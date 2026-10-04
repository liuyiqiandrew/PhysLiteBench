"""Validate harmonic calibration and Oldroyd-B sheet pumping."""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.integrate import quad

STAGE = Path(__file__).resolve().parents[1]
TASK = STAGE/'tasks/waving-sheet'
TRUE = 1.05
SIGMA = .002
SEED = 285021
NOISE_SEED = 285027
CHANNELS = ['u_real','u_imag','v_real','v_imag']
DIAGNOSTICS = ['frequency_scan','wavelength_scan','combined_controls']


def load(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle = load('sheet_oracle',TASK/'solution/model.py')
shortcut = load('sheet_shortcut',STAGE/'scripts/waving_sheet_baseline.py')
reference = load('sheet_reference',TASK/'tests/reference.py')
prototype = load('sheet_prototype',STAGE/'prototype/prototype.py')


def calibration_inputs():
    return [reference.experiment(k,w,o,h) for k,w,h,o in itertools.product(
        [.7,1.3],[.5,2.,8.],[.5,.6],CHANNELS)]*6


def errors(model,truth):
    return {name:float(np.linalg.norm(model.predict(inputs)-truth[name])/np.linalg.norm(truth[name]))
            for name,inputs in reference.hidden_inputs().items()}


def evaluate(records,truth):
    inputs = [r['input'] for r in records]
    y = np.array([r['value'] for r in records])
    sigma = np.array([r['sigma'] for r in records])
    result = {}
    for name,module in [('oracle',oracle),('shortcut',shortcut)]:
        model = module.Model().fit(records)
        residual = (model.predict(inputs)-y)/sigma
        result[name] = {'viscosity':model.viscosity,
            'calibration_chi2':float(residual@residual/(len(y)-1)),
            'parameter_relative_error':abs(model.viscosity/TRUE-1),'hidden':errors(model,truth)}
    return result


def physical_checks(nu,k,w):
    result = reference.boundary_reference(nu,k,w)
    raw = result['linear_state'](0.)
    state = raw[:4]+1j*raw[4:]
    viscosity = reference.complex_viscosity(nu,w)
    pressure = (1j*w*state[1]+viscosity*(state[3]-k*k*state[1]))/(1j*k)
    vy = -1j*k*state[1]
    wall_velocity = -1j*k*state[0]
    power = .5*np.real(wall_velocity*np.conj(pressure-2*viscosity*vy))
    def dissipation(y):
        psi = prototype.derivatives(nu,k,w,y)
        u,second = psi[1],psi[2]
        return viscosity.real*(2*k*k*abs(u)**2+.5*abs(second+k*k*psi[0])**2)
    dissipated = quad(dissipation,0,np.inf,epsabs=1e-10,epsrel=1e-10)[0]
    y = np.linspace(0,min(result['height'],15/k),101)
    u,v = result['velocity'](y)
    momentum = nu*result['mean_state'](y)[1]+result['nonlinear_stress'](y)-.5*np.real(u*np.conj(v))
    return {'wall_power_per_density':float(power),'dissipation_per_density':float(dissipated),
        'energy_relative_error':float(abs(power/dissipated-1)),
        'momentum_flux_residual':float(np.max(abs(momentum))/max(1,abs(result['physical']*nu*k))),
        'wall_normal_velocity_error':float(abs(wall_velocity+1j*w)),
        'wall_tangential_velocity_error':float(abs(state[1]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate',action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    inputs = calibration_inputs()
    clean = reference.predict(inputs,TRUE)
    path = TASK/'environment/data/calibration.json'
    if args.generate:
        rng = np.random.default_rng(SEED)
        records = [{'input':e,'value':float(v),'sigma':SIGMA}
                   for e,v in zip(inputs,clean+SIGMA*rng.normal(size=len(inputs)))]
        text = json.dumps(records,indent=2)+'\n'
        path.write_text(text)
        (TASK/'tests/data/calibration.json').write_text(text)
    records = json.loads(path.read_text())
    assert [r['input'] for r in records]==inputs
    assert all(set(r)=={'input','value','sigma'} and r['sigma']==SIGMA for r in records)
    assert path.read_bytes()==(TASK/'tests/data/calibration.json').read_bytes()
    cold = time.perf_counter()
    truth = {k:reference.predict(v,TRUE) for k,v in reference.hidden_inputs().items()}
    reference_seconds = time.perf_counter()-cold
    nominal = evaluate(records,truth)
    rng = np.random.default_rng(NOISE_SEED)
    fits = []
    for _ in range(256):
        draw = [{'input':e,'value':float(v),'sigma':SIGMA}
                for e,v in zip(inputs,clean+SIGMA*rng.normal(size=len(inputs)))]
        value = evaluate(draw,truth)
        assert value['oracle']['viscosity']==value['shortcut']['viscosity']
        for control in value.values():
            assert control['calibration_chi2']<1.5
            assert control['parameter_relative_error']<.03
        assert max(value['oracle']['hidden'].values())<.04
        assert all(value['shortcut']['hidden'][k]>.04 for k in DIAGNOSTICS)
        assert value['shortcut']['hidden']['linear_anchors']<.04
        fits.append(value)

    cases = list(itertools.product([.7,1.4],[.7,1.3],[.5,2.,8.]))
    cases += [(rng.uniform(.7,1.4),rng.uniform(.7,1.3),rng.uniform(.5,8)) for _ in range(12)]
    domain = []
    equivalence = identity = source_match = 0.
    for nu,k,w in cases:
        e = reference.experiment(k,w)
        physical = oracle.predict_at([e],nu)[0]
        source = shortcut.predict_at([e],nu)[0]
        ref = reference.boundary_reference(nu,k,w)
        linear = [reference.experiment(k,w,o,h) for h,o in itertools.product([.2,2.],CHANNELS)]
        exact = oracle.predict_at(linear,nu)
        independent = reference.predict(linear,nu)
        equivalence = max(equivalence,float(np.max(abs(exact-shortcut.predict_at(linear,nu)))))
        source_match = max(source_match,abs(source/ref['source']-1))
        check = physical_checks(nu,k,w)
        row = {'viscosity':nu,'wave_number':k,'frequency':w,'oracle':float(physical),
            'shortcut':float(source),'reference':ref['physical'],
            'pumping_relative_error':abs(physical/ref['physical']-1),
            'linear_error':float(np.max(abs(exact-independent)/np.maximum(1,abs(exact)))),**check}
        assert physical>0 and np.isfinite(source) and check['wall_power_per_density']>0
        assert row['pumping_relative_error']<1e-6 and row['linear_error']<1e-7
        assert check['energy_relative_error']<1e-6 and check['momentum_flux_residual']<1e-6
        for h in [.2,1.,2.]:
            u,v = prototype.velocity(nu,k,w,h)
            s = np.sqrt(k*k-1j*w/prototype.complex_viscosity(nu,w))
            identity = max(identity,float(abs((u+1j*v)/(w*np.exp(-s*h))-1)))
        domain.append(row)
    assert equivalence==0 and identity<1e-11 and source_match<1e-6

    refinement = []
    for nu,k,w in [(.7,.7,8.),(1.4,1.3,.5),(1.05,1.,4.)]:
        a = reference.boundary_reference(nu,k,w)
        b = reference.boundary_reference(nu,k,w,depth=30.,tolerance=2e-9)
        refinement.append(abs(a['physical']/b['physical']-1))
    assert max(refinement)<1e-7

    grid = np.linspace(.7,1.4,401)
    predictions = np.array([oracle.predict_at(inputs,nu) for nu in grid])
    recovery = []
    gaps = []
    for nu in np.linspace(.7,1.4,25):
        y = oracle.predict_at(inputs,nu)
        noise_free = [{'input':e,'value':float(v),'sigma':SIGMA} for e,v in zip(inputs,y)]
        fitted = oracle.Model().fit(noise_free).viscosity
        objective = np.sum((predictions-y)**2,axis=1)
        local = int(np.sum((objective[1:-1]<objective[:-2])&(objective[1:-1]<objective[2:])))
        assert abs(fitted/nu-1)<1e-7
        assert local==(0 if nu in [.7,1.4] else 1)
        recovery.append({'true':float(nu),'fit':fitted,'relative_error':abs(fitted/nu-1),
                         'interior_grid_minima':local})
        m = shortcut.Model();m.viscosity=float(nu)
        truth_at_nu = {k:oracle.predict_at(v,nu) for k,v in reference.hidden_inputs().items()}
        gaps.append({'viscosity':float(nu),'hidden':errors(m,truth_at_nu)})
    # The full complex two-height ratio identifies s without a phase branch.
    phase_bound = .1*np.sqrt(1.3**2+8/(.25*.7))
    assert phase_bound<np.pi
    ratio_recovery = []
    for nu,k,w in cases:
        q=[]
        for h in [.5,.6]:
            u,v = prototype.velocity(nu,k,w,h)
            q.append(u+1j*v)
        s_value = -np.log(q[1]/q[0])/.1
        factor = .25+.75/(1-1j*w)
        recovered = -1j*w/((s_value*s_value-k*k)*factor)
        ratio_recovery.append(abs(recovered/nu-1))
    assert max(ratio_recovery)<1e-11

    limits=[]
    for nu in [10.,100.,1000.,10000.]:
        result=prototype.integrated_prediction(nu,1.,2.)
        limits.append({'viscosity':nu,**result,'ratio':result['physical']/result['source']})
    assert abs(limits[-1]['ratio']/.4-1)<.001
    newtonian=[]
    for beta,relaxation in [(1.,1.),(.25,0.)]:
        r=prototype.integrated_prediction(1.05,1.,2.,beta,relaxation)
        assert r['polymer']==0
        newtonian.append(dict(beta=beta,relaxation=relaxation,**r))
    reversal=prototype_match=0.
    for nu,k,w in cases:
        e=reference.experiment(k,w)
        actual=oracle.predict_at([e],nu)[0]
        reversal=max(reversal,abs(oracle.predict_at([dict(e,frequency=-w)],nu)[0]+actual))
        p=prototype.integrated_prediction(nu,k,w)
        prototype_match=max(prototype_match,abs(actual/p['physical']-1),
            abs(shortcut.predict_at([e],nu)[0]/p['source']-1))
    assert reversal<1e-10 and prototype_match<1e-9
    assert oracle.predict_at([],TRUE).shape==(0,)
    all_controls = [v[k] for v in fits for k in ['oracle','shortcut']]
    report = {
        'task':'waving-sheet','revision':2,'status':'scientific_validation_passed_no_model_evaluation',
        'nominal':nominal,
        'data_integrity':{'record_count':len(records),
            'distinct_calibration_settings':len({json.dumps(e,sort_keys=True) for e in inputs}),
            'repetitions':6,'fixed_sigma':SIGMA,'sigma_depends_on_response_or_unknown':False,
            'public_private_identical':True,'calibration_seed':SEED,'noise_seed':NOISE_SEED,
            'independent_calibration_bias_in_sigma':float(np.max(abs(clean-oracle.predict_at(inputs,TRUE)))/SIGMA)},
        'noise':{'draws':256,'oracle_passes':256,'shortcut_rejections':256,
            'max_calibration_chi2':max(v['calibration_chi2'] for v in all_controls),
            'max_parameter_relative_error':max(v['parameter_relative_error'] for v in all_controls),
            'max_oracle_hidden_error':max(max(v['oracle']['hidden'].values()) for v in fits),
            'minimum_shortcut_diagnostic_error':min(v['shortcut']['hidden'][k] for v in fits for k in DIAGNOSTICS)},
        'domain':{'case_count':len(domain),'corner_count':12,'random_count':12,'cases':domain,
            'max_reference_relative_error':max(v['pumping_relative_error'] for v in domain),
            'max_linear_error':max(v['linear_error'] for v in domain),
            'max_energy_relative_error':max(v['energy_relative_error'] for v in domain),
            'max_momentum_flux_residual':max(v['momentum_flux_residual'] for v in domain)},
        'reference_refinement_relative_change_max':max(refinement),
        'calibration_exact_equivalence_error':equivalence,'source_independent_boundary_error':source_match,
        'calibration_injective_identity_error':identity,
        'two_height_phase_bound_radians':float(phase_bound),
        'two_height_parameter_recovery_error':float(max(ratio_recovery)),
        'noiseless_calibration_recovery':recovery,'full_parameter_range_gaps':gaps,
        'minimum_full_range_shortcut_error':min(v['hidden'][k] for v in gaps for k in DIAGNOSTICS),
        'creeping_limits':limits,'newtonian_limits':newtonian,
        'pumping_reversal_error':reversal,'preserved_prototype_agreement':prototype_match,
        'prototype_limit_qualification':'Optional beta/lambda limits are checked in the preserved generalized prototype; task inputs keep beta=.25 and lambda=1. Current task predictions independently match the prototype throughout the sampled public domain.',
        'broad_domain_gap_qualification':'Accidental near-equivalence outside the scored high-frequency groups is preserved in the prototype report; no all-domain separation claim.',
        'minimum_hidden_pumping':min(float(np.min(truth[k])) for k in DIAGNOSTICS),
        'cold_hidden_reference_seconds':reference_seconds,'seconds':time.perf_counter()-started,
        'source_sha256':{str(p.relative_to(STAGE)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted([*TASK.rglob('*'),STAGE/'scripts/waving_sheet_baseline.py',Path(__file__)])
            if p.is_file() and '__pycache__' not in str(p)},
    }
    (STAGE/'results/waving-sheet-r2-validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in
                     ['source_sha256','domain','noiseless_calibration_recovery','full_parameter_range_gaps']},indent=2))


if __name__=='__main__':
    main()
