"""Validate radial capture and survival-conditioned accumulated transport."""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigh_tridiagonal

STAGE = Path(__file__).resolve().parents[1]
TASK = STAGE/'tasks/survivor-transport'
TRUE = 1.05
SIGMA = {'loss_rate':.005,'drift':.002}
SEED = 271021
NOISE_SEED = 271027
DIAGNOSTICS = ['capture_scan','geometry_scan','reverse_flow']


def load(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle = load('survivor_oracle',TASK/'solution/model.py')
shortcut = load('survivor_shortcut',STAGE/'scripts/survivor_transport_baseline.py')
reference = load('survivor_reference',TASK/'tests/reference.py')
prototype = load('survivor_prototype',STAGE/'prototype/prototype.py')


def calibration_inputs():
    inputs = [reference.experiment(r,k,observable='loss_rate')
              for r,k in itertools.product([.7,1.,1.3],[.5,2.,8.])]*16
    inputs += [reference.experiment(r,k,plug=u)
               for r,k,u in itertools.product([.8,1.2],[1.,12.],[-.6,.9])]*12
    inputs += [reference.experiment(r,0.,peak=u)
               for r,u in itertools.product([.8,1.2],[-1.2,.8])]*12
    return inputs


def errors(model,truth):
    return {name:float(np.linalg.norm(model.predict(inputs)-truth[name])/
                       np.linalg.norm(truth[name]))
            for name,inputs in reference.hidden_inputs().items()}


def evaluate(records,truth):
    inputs = [r['input'] for r in records]
    y = np.array([r['value'] for r in records])
    sigma = np.array([r['sigma'] for r in records])
    result = {}
    for name,module in [('oracle',oracle),('shortcut',shortcut)]:
        model = module.Model().fit(records)
        residual = (model.predict(inputs)-y)/sigma
        result[name] = {'diffusivity':model.diffusivity,
            'calibration_chi2':float(residual@residual/(len(y)-1)),
            'parameter_relative_error':abs(model.diffusivity/TRUE-1),
            'hidden':errors(model,truth)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate',action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    inputs = calibration_inputs()
    clean = reference.predict(inputs,TRUE)
    instrument = np.array([SIGMA[e['observable']] for e in inputs])
    path = TASK/'environment/data/calibration.json'
    if args.generate:
        rng = np.random.default_rng(SEED)
        records = [{'input':e,'value':float(v),'sigma':float(s)}
                   for e,v,s in zip(inputs,clean+instrument*rng.normal(size=len(inputs)),instrument)]
        data = json.dumps(records,indent=2)+'\n'
        path.write_text(data)
        (TASK/'tests/data/calibration.json').write_text(data)
    records = json.loads(path.read_text())
    assert [r['input'] for r in records]==inputs
    assert all(set(r)=={'input','value','sigma'} and r['sigma']==SIGMA[r['input']['observable']] for r in records)
    assert path.read_bytes()==(TASK/'tests/data/calibration.json').read_bytes()
    cold = time.perf_counter()
    truth = {k:reference.predict(v,TRUE) for k,v in reference.hidden_inputs().items()}
    reference_seconds = time.perf_counter()-cold
    nominal = evaluate(records,truth)

    rng = np.random.default_rng(NOISE_SEED)
    fits = []
    for _ in range(256):
        draw = [{'input':e,'value':float(v),'sigma':float(s)}
                for e,v,s in zip(inputs,clean+instrument*rng.normal(size=len(inputs)),instrument)]
        value = evaluate(draw,truth)
        assert value['oracle']['diffusivity']==value['shortcut']['diffusivity']
        for control in value.values():
            assert control['calibration_chi2']<1.5
            assert control['parameter_relative_error']<.03
        assert max(value['oracle']['hidden'].values())<.04
        assert all(value['shortcut']['hidden'][k]>.04 for k in DIAGNOSTICS)
        assert value['shortcut']['hidden']['shared_anchors']<.04
        fits.append(value)

    corners = list(itertools.product([.7,1.4],[.7,1.3],[0.,.5,8.,24.],[-1.3,1.3]))
    cases = [(d,r,k,0.,u) for d,r,k,u in corners]
    cases += [(rng.uniform(.7,1.4),rng.uniform(.7,1.3),rng.uniform(0,24),
               rng.uniform(-1,1),rng.uniform(-1.5,1.5)) for _ in range(32)]
    domain = []
    for d,r,k,plug,peak in cases:
        experiments = [reference.experiment(r,k,plug,peak,o) for o in ['loss_rate','drift']]
        physical = oracle.predict_at(experiments,d)
        source = shortcut.predict_at(experiments,d)
        ref = reference.predict(experiments,d)
        q = oracle.radial_mode(d,r,k)
        beta = k*r/d
        profile = oracle.j0(q*oracle.R)
        velocity_bounds = sorted([plug,plug+peak])
        assert 0<=q<oracle.EDGE or (k==0 and q==0)
        assert physical[0]>=0 and source[0]>=0 and np.min(profile)>0
        assert velocity_bounds[0]-1e-12<=physical[1]<=velocity_bounds[1]+1e-12
        assert velocity_bounds[0]-1e-12<=source[1]<=velocity_bounds[1]+1e-12
        root_residual = abs(q*oracle.j1(q)-beta*oracle.j0(q))
        row = {'diffusivity':d,'radius':r,'capture':k,'plug':plug,'peak':peak,
               'oracle':physical.tolist(),'shortcut':source.tolist(),'reference':ref.tolist(),
               'reference_error':float(np.max(abs(physical-ref)/np.maximum(1,abs(physical)))),
               'boundary_residual':root_residual,'minimum_mode':float(profile.min())}
        assert row['reference_error']<1e-6
        assert root_residual<1e-11
        domain.append(row)

    refinements = []
    finite_time = []
    mass_balance = []
    for r,k,peak in [(1.,6.,1.),(.75,16.,.8),(1.3,24.,-1.4)]:
        coarse = reference.response(TRUE,r,k,0.,peak,256)
        fine = reference.response(TRUE,r,k,0.,peak,512)
        refinements.append(float(np.max(abs(fine-coarse)/np.maximum(1,abs(fine)))))
        centers,volume,diagonal,off = reference.operator(TRUE,r,k,128)
        eigen,vectors = eigh_tridiagonal(diagonal,off,select='i',select_range=(126,127))
        density = vectors[:,-1]/np.sqrt(volume)
        if density.sum()<0:
            density = -density
        rate = -float(eigen[-1])
        dr = r/128
        wall_density = density[-1]/(1+k*dr/(2*TRUE))
        balance = abs(rate*np.dot(volume,density)-r*k*wall_density)/max(1,rate*np.dot(volume,density))
        mass_balance.append(float(balance))
        gap = float(eigen[-1]-eigen[-2])
        times = np.array([2.,4.,8.,16.,32.])/gap
        means = prototype.conditional_moment(TRUE,r,k,0.,peak,times,128)
        slopes = np.diff(means)/np.diff(times)
        target = oracle.predict_at([reference.experiment(r,k,peak=peak)],TRUE)[0]
        finite_time.append({'radius':r,'capture':k,'peak':peak,'loss_rate':rate,'gap':gap,
            'times':times.tolist(),'conditional_means':means,'interval_slopes':slopes.tolist(),
            'last_slope_relative_error':abs(slopes[-1]/target-1)})
    assert max(refinements)<1e-6
    assert max(mass_balance)<1e-9
    assert max(x['last_slope_relative_error'] for x in finite_time)<2e-5

    equivalence = reversal = shift = prototype_match = 0.
    for d in [.7,.9,TRUE,1.4]:
        equivalence = max(equivalence,float(np.max(abs(oracle.predict_at(inputs,d)-shortcut.predict_at(inputs,d)))))
        for r,k,peak in [(.7,0.,1.5),(1.,8.,-.8),(1.3,24.,1.2)]:
            e = reference.experiment(r,k,peak=peak)
            pos = oracle.predict_at([e],d)[0]
            reversal = max(reversal,abs(pos+oracle.predict_at([dict(e,peak=-peak)],d)[0]))
            shift = max(shift,abs(oracle.predict_at([dict(e,plug=.7)],d)[0]-pos-.7))
            old = prototype.analytic(d,r,k,0.,peak)
            prototype_match = max(prototype_match,abs(shortcut.predict_at([e],d)[0]-old['source_drift']),
                                  abs(pos-old['physical_drift']))
    assert equivalence==0
    assert max(reversal,shift,prototype_match)<1e-12
    assert oracle.predict_at([],TRUE).shape==(0,)

    grid = np.linspace(.7,1.4,501)
    grid_prediction = np.array([oracle.predict_at(inputs,d) for d in grid])
    rate_indices = [i for i,e in enumerate(inputs) if e['observable']=='loss_rate']
    sensitivities = np.diff(grid_prediction[:,rate_indices],axis=0)/np.diff(grid)[:,None]
    assert sensitivities.min()>0
    recovery = []
    full_range_gaps = []
    for d in np.linspace(.7,1.4,25):
        y = oracle.predict_at(inputs,d)
        records_at_d = [{'input':e,'value':float(v),'sigma':float(s)} for e,v,s in zip(inputs,y,instrument)]
        fitted = oracle.Model().fit(records_at_d).diffusivity
        objective = np.sum(((grid_prediction-y)/instrument)**2,axis=1)
        minima = int(np.sum((objective[1:-1]<objective[:-2])&(objective[1:-1]<objective[2:])))
        assert abs(fitted/d-1)<1e-7
        assert minima==(0 if d in [.7,1.4] else 1)
        recovery.append({'true':float(d),'fit':fitted,'relative_error':abs(fitted/d-1),
                         'interior_grid_minima':minima})
        m = shortcut.Model();m.diffusivity=float(d)
        truth_at_d = {name:oracle.predict_at(e,d) for name,e in reference.hidden_inputs().items()}
        full_range_gaps.append({'diffusivity':float(d),'hidden':errors(m,truth_at_d)})
    assert min(v['hidden'][k] for v in full_range_gaps for k in DIAGNOSTICS)>.04

    all_controls = [v[k] for v in fits for k in ['oracle','shortcut']]
    report = {
        'task':'survivor-transport','revision':1,
        'status':'scientific_validation_passed_no_model_evaluation','nominal':nominal,
        'data_integrity':{'record_count':len(records),
            'distinct_calibration_settings':len({json.dumps(e,sort_keys=True) for e in inputs}),
            'loss_settings':9,'loss_repetitions':16,'drift_settings':12,'drift_repetitions':12,
            'fixed_sigma':SIGMA,'sigma_depends_on_response_or_unknown':False,
            'public_private_identical':True,'calibration_seed':SEED,'noise_seed':NOISE_SEED,
            'independent_calibration_bias_in_sigma':float(np.max(abs(clean-oracle.predict_at(inputs,TRUE))/instrument))},
        'noise':{'draws':256,'oracle_passes':256,'shortcut_rejections':256,
            'max_calibration_chi2':max(v['calibration_chi2'] for v in all_controls),
            'max_parameter_relative_error':max(v['parameter_relative_error'] for v in all_controls),
            'max_oracle_hidden_error':max(max(v['oracle']['hidden'].values()) for v in fits),
            'minimum_shortcut_diagnostic_error':min(v['shortcut']['hidden'][k] for v in fits for k in DIAGNOSTICS)},
        'domain':{'case_count':len(domain),'cases':domain,
            'maximum_reference_error':max(v['reference_error'] for v in domain)},
        'refinement_change_max':max(refinements),'reference_mass_balance_error':max(mass_balance),
        'finite_time_conditional_convergence':finite_time,'identifiability':recovery,
        'minimum_calibration_derivative':float(sensitivities.min()),
        'full_parameter_range_gaps':full_range_gaps,
        'minimum_full_range_shortcut_error':min(v['hidden'][k] for v in full_range_gaps for k in DIAGNOSTICS),
        'calibration_exact_equivalence_error':equivalence,'flow_reversal_error':reversal,
        'uniform_velocity_shift_error':shift,'frozen_prototype_match_error':prototype_match,
        'minimum_hidden_absolute_drift':min(float(np.min(abs(truth[k]))) for k in DIAGNOSTICS),
        'cold_hidden_reference_seconds':reference_seconds,'seconds':time.perf_counter()-started,
        'source_sha256':{str(p.relative_to(STAGE)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted([*TASK.rglob('*'),STAGE/'scripts/survivor_transport_baseline.py',Path(__file__)])
            if p.is_file() and '__pycache__' not in str(p)},
    }
    output = STAGE/'results/survivor-transport-r1-validation.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in
                     ['source_sha256','domain','identifiability','full_parameter_range_gaps','finite_time_conditional_convergence']},indent=2))


if __name__=='__main__':
    main()
