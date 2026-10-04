"""Validate Navier-slip force transmission with independent solute and Stokes fields."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import itertools
import json
import time
import numpy as np
from scipy.integrate import quad, solve_bvp

BASE = Path(__file__).resolve().parents[1]
TASK = BASE/'tasks/finite-layer-phoresis'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalized(actual, truth):
    return float(np.sqrt(np.mean((actual-truth)**2)/np.mean(truth**2)))


def force_check(e, reference):
    a,w,s,b = (e[k] for k in ['radius','width','strength','slip'])
    end = a+w
    c = reference.concentration(a,w,s,512)
    derivative = lambda r: -2*s*(end-r)/w**2
    def rhs(r,y):
        return np.array([y[1],y[2],y[3],-c(r)*derivative(r)+4*y[2]/r**2-8*y[1]/r**3+8*y[0]/r**4])
    def bc(l,r):
        return np.array([l[0],(1+2*b/a)*l[1]-b*l[2],r[2]-2*r[0]/end**2,
                         r[3]-2*r[1]/end**2+4*r[0]/end**3])
    grid = np.linspace(a,end,257)
    sol = solve_bvp(rhs,bc,grid,np.zeros((4,len(grid))),tol=2e-10,max_nodes=20000)
    assert sol.success
    wall,outer = sol.sol(a),sol.sol(end)
    fluid = 4*np.pi*a*a/3*(-wall[3]+2*wall[2]/a+2*wall[1]/a**2)
    reaction = 4*np.pi/3*quad(lambda r:r*r*c(r)*derivative(r),a,end,epsabs=2e-11,epsrel=2e-11)[0]
    return {'force_balance':abs(fluid+reaction)/max(1.,abs(fluid),abs(reaction)),
            'boundary_residual':float(np.max(abs(bc(wall,outer)))),
            'slip_dissipation':float(8*np.pi/3*wall[1]**2/b) if b else 0.}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate',action='store_true')
    args = parser.parse_args()
    start = time.monotonic()
    oracle = load(TASK/'solution/model.py','oracle')
    shortcut = load(BASE/'scripts/finite_layer_phoresis_baseline.py','shortcut')
    reference = load(TASK/'tests/reference.py','reference')
    metadata = json.loads((TASK/'tests/metadata.json').read_text())
    sigma,true = metadata['sigma'],reference.TRUE_PARAMETER
    inputs = reference.calibration_inputs()
    noiseless = reference.predict(inputs)
    if args.generate:
        rng = np.random.default_rng(metadata['calibration_seed'])
        records = [dict(input=e,value=float(v),sigma=sigma)
                   for e,v in zip(inputs,noiseless+rng.normal(0,sigma,len(inputs)))]
        data = json.dumps(records,indent=2)+'\n'
        for path in [TASK/'environment/data/calibration.json',TASK/'tests/data/calibration.json']:
            path.write_text(data)
    records = json.loads((TASK/'tests/data/calibration.json').read_text())
    assert records == json.loads((TASK/'environment/data/calibration.json').read_text())
    assert [r['input'] for r in records] == inputs
    assert all(set(r)=={'input','value','sigma'} and r['sigma']==sigma for r in records)
    assert sigma == 3e-5 and metadata['prediction_limit']==.04
    hidden = reference.hidden_inputs()
    truth = {key:reference.predict(es) for key,es in hidden.items()}
    nominal = {}
    for name,module in [('oracle',oracle),('shortcut',shortcut)]:
        m = module.Model().fit(records)
        residual = (m.predict(inputs)-np.array([r['value'] for r in records]))/sigma
        nominal[name] = {'viscosity':m.viscosity,'parameter_relative_error':abs(m.viscosity/true-1),
                         'calibration_chi2':float(residual@residual)/(len(inputs)-1),
                         'hidden':{key:normalized(m.predict(es),truth[key]) for key,es in hidden.items()}}
    assert max(nominal['oracle']['hidden'].values())<.04
    assert min(v for k,v in nominal['shortcut']['hidden'].items() if k!='no_slip_anchors')>.04
    rng = np.random.default_rng(metadata['noise_validation_seed'])
    noise = []
    for _ in range(256):
        y = noiseless+rng.normal(0,sigma,len(inputs))
        rr = [dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,y)]
        results = {}
        for name,module in [('oracle',oracle),('shortcut',shortcut)]:
            m = module.Model().fit(rr)
            chi = float(np.sum(((m.predict(inputs)-y)/sigma)**2)/(len(inputs)-1))
            errors = {key:normalized(m.predict(es),truth[key]) for key,es in hidden.items()}
            assert abs(m.viscosity/true-1)<.03 and chi<1.5
            if name=='oracle':assert max(errors.values())<.04
            else:assert min(v for k,v in errors.items() if k!='no_slip_anchors')>.04
            results[name] = dict(viscosity=m.viscosity,chi2=chi,hidden=errors)
        noise.append(results)
    corners = [reference.experiment(a,w,s,b) for a,w,s,b in itertools.product(
        [.7,1.3],[.05,.5,1.7,3.5],[-2.,-.5,.5,2.],[0.,.4,2.])]
    corners += [reference.experiment(.91,2.37,1.13,.62),reference.experiment(1.14,2.62,-1.37,1.77)]
    ref = reference.predict(corners)
    actual = oracle.predict_at(corners,true)
    refined = reference.predict(corners,cells=384)
    error = float(np.max(abs(actual-ref)))
    refinement = float(np.max(abs(refined-ref)))
    assert error<2e-7 and refinement<2e-7
    checks = [force_check(reference.experiment(*x),reference) for x in [
        (.7,.05,2.,2.),(1.3,3.5,-2.,2.),(.9,.8,.8,.3),(1.1,2.3,-1.2,0.)]]
    assert max(x['force_balance'] for x in checks)<2e-8
    assert max(x['boundary_residual'] for x in checks)<2e-8
    assert all(x['slip_dissipation']>=0 for x in checks)
    zero = [reference.experiment(.7,.05,0.,0.),reference.experiment(1.3,3.5,0.,2.)]
    assert np.max(abs(oracle.predict_at(zero,true)))==0
    assert np.max(abs(shortcut.predict_at(zero,true)))==0
    assert np.max(abs(reference.predict(zero)))==0
    scaling = []
    for module in [oracle,shortcut]:
        e = reference.experiment(.8,.56,1.1,.64)
        larger = reference.experiment(1.2,.84,1.1,.96)
        scaling.append(abs(module.predict_at([larger],1.)[0]/module.predict_at([e],1.)[0]-2.25))
    assert max(scaling)<1e-8
    width,slip,strength = .0003,.0003,1.2
    planar = -quad(lambda z:(z+slip)*np.expm1(-strength*(1-z/width)**2),0,width,
                   epsabs=1e-17,epsrel=1e-11)[0]
    thin = oracle.response(1.,width,strength,slip)/(-planar)
    assert abs(thin-1)<.001
    exact_recovery = []
    for eta in [.8,1.,1.2,1.6]:
        rr = [dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,oracle.predict_at(inputs,eta))]
        for module in [oracle,shortcut]:
            exact_recovery.append(abs(module.Model().fit(rr).viscosity/eta-1))
    assert max(exact_recovery)<1e-12
    equivalence = float(np.max(abs(oracle.predict_at(inputs,1.)-shortcut.predict_at(inputs,1.))))
    assert equivalence<1e-13
    # The parameter enters only as 1/eta; positive design norm proves unique WLS fitting.
    inverse_viscosity_curvature = float(2*np.sum((oracle.predict_at(inputs,1.)/sigma)**2))
    assert inverse_viscosity_curvature>0
    groups = [key for key in hidden if key!='no_slip_anchors']
    report = {'task':'finite-layer-phoresis','revision':2,'noise_trials':256,'controls':nominal,
      'history':{'r1':'Archived 1/3 passes, two reviewed physical failures; all outcomes retained.',
                 'r2':'Navier boundary and spherical no-slip calibration; no model evaluations yet.'},
      'noise':{'oracle_passes':256,'shortcut_rejections':256,'all_calibration_pass':True,
        'max_calibration_chi2':max(x[n]['chi2'] for x in noise for n in x),
        'max_parameter_relative_error':max(abs(x[n]['viscosity']/true-1) for x in noise for n in x),
        'max_oracle_hidden_error':max(v for x in noise for v in x['oracle']['hidden'].values()),
        'min_shortcut_slip_error':min(x['shortcut']['hidden'][k] for x in noise for k in groups)},
      'physical_checks':{'independent_reference_cases':len(corners),'oracle_reference_error_max':error,
        'reference_refinement_change_max':refinement,'force_balance_cases':checks,
        'geometric_scaling_error_max':max(scaling),'thin_sphere_to_opposite_planar_slip_ratio':thin,
        'zero_interaction_exact':True,'noiseless_parameter_recovery_error_max':max(exact_recovery),
        'calibration_equivalence_max':equivalence,'inverse_viscosity_objective_curvature':inverse_viscosity_curvature,
        'minimum_hidden_slip_signal':min(float(np.min(abs(truth[k]))) for k in groups),
        'maximum_hidden_slip_signal':max(float(np.max(abs(truth[k]))) for k in groups),
        'positive_drag_factor_domain':[2/3,1.],
        'kernel_wall_value':0.,'kernel_wall_derivative':'3*a*(b/a)/(1+3*b/a)'},
      'data_integrity':{'records':len(records),'distinct_calibration_settings':len({json.dumps(e,sort_keys=True) for e in inputs}),'fixed_instrument_sigma':sigma,
        'public_private_identical':True,'noiseless_sigma_side_channel':False,
        'seeds':{'calibration':metadata['calibration_seed'],'noise':metadata['noise_validation_seed']}},
      'runtime_seconds':time.monotonic()-start,
      'source_sha256':{str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(TASK.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}}
    out = BASE/'results/finite-layer-phoresis-r2-validation.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))


if __name__=='__main__':
    main()
