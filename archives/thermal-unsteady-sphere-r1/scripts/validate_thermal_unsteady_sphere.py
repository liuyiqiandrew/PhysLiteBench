"""Validate driven calibration and equilibrium sphere spectra."""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize_scalar

STAGE = Path(__file__).resolve().parents[1]
TASK = STAGE/'tasks/thermal-unsteady-sphere'
TRUE = 1.07
SIGMA = .002
SEED = 261017
NOISE_SEED = 261019


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle = load('sphere_oracle', TASK/'solution/model.py')
shortcut = load('sphere_shortcut', STAGE/'scripts/thermal_unsteady_sphere_baseline.py')
reference = load('sphere_reference', TASK/'tests/reference.py')


def calibration_inputs():
    inputs = []
    for radius, frequency in itertools.product([.75,1.25], [0.,.2,.8,2.]):
        inputs.append(reference.experiment(radius,frequency,300.,'in_phase'))
        if frequency:
            inputs.append(reference.experiment(radius,frequency,300.,'quadrature'))
    return inputs*16


def errors(model, truth):
    return {name:float(np.linalg.norm(model.predict(inputs)-truth[name])/
                       np.linalg.norm(truth[name]))
            for name,inputs in reference.hidden_inputs().items()}


def evaluate(records, truth):
    inputs = [r['input'] for r in records]
    y = np.array([r['value'] for r in records])
    sigma = np.array([r['sigma'] for r in records])
    result = {}
    for name, module in [('oracle',oracle),('shortcut',shortcut)]:
        model = module.Model().fit(records)
        residual = (model.predict(inputs)-y)/sigma
        result[name] = {
            'viscosity':model.viscosity,
            'calibration_chi2':float(residual@residual/(len(y)-1)),
            'parameter_relative_error':abs(model.viscosity/TRUE-1),
            'hidden':errors(model,truth),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate',action='store_true')
    args = parser.parse_args()
    start = time.perf_counter()
    inputs = calibration_inputs()
    noiseless = oracle.predict_at(inputs, TRUE)
    path = TASK/'environment/data/calibration.json'
    if args.generate:
        rng = np.random.default_rng(SEED)
        values = noiseless+SIGMA*rng.normal(size=len(inputs))
        records = [{'input':e,'value':float(v),'sigma':SIGMA}
                   for e,v in zip(inputs,values)]
        data = json.dumps(records,indent=2)+'\n'
        path.write_text(data)
        (TASK/'tests/data/calibration.json').write_text(data)
    records = json.loads(path.read_text())
    assert records and len(records)==len(inputs)
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
                for e,v in zip(inputs,noiseless+SIGMA*rng.normal(size=len(inputs)))]
        value = evaluate(draw,truth)
        assert value['oracle']['viscosity']==value['shortcut']['viscosity']
        for control in value.values():
            assert control['calibration_chi2']<1.5
            assert control['parameter_relative_error']<.03
        assert max(value['oracle']['hidden'].values())<.04
        assert all(value['shortcut']['hidden'][k]>.04 for k in
                   ['frequency_scan','radius_scan','temperature_controls'])
        fits.append(value)

    cases = list(itertools.product([.5,1.5], [0.,.2,8.], [.7,1.4], [280.,320.]))
    cases += [(rng.uniform(.5,1.5),rng.uniform(.2,8),rng.uniform(.7,1.4),
               rng.uniform(280,320)) for _ in range(32)]
    domain = []
    for radius,frequency,eta,temp in cases:
        e = reference.experiment(radius,frequency,temp)
        physical = oracle.predict_at([e],eta)[0]
        alternate = shortcut.predict_at([e],eta)[0]
        ref = reference.fluid_reference(radius*1e-6,frequency*1e6,eta*1e-3,temp)
        mu = oracle.mobility(radius,frequency,eta)
        mass = 4*np.pi*2200*(radius*1e-6)**3/3
        impedance = 1/mu+1j*frequency*1e6*mass
        row = {
            'input':e,'viscosity':eta,
            'oracle_psd':float(physical),'shortcut_psd':float(alternate),
            'oracle_reference_relative_error':float(abs(physical-ref['spectrum']*1e12)/physical),
            'mobility_reference_relative_error':float(abs(mu-ref['mobility'])/abs(mu)),
            'dissipation_relative_error':float(abs(ref['dissipation_integral']-impedance.real)/impedance.real),
            'boundary_error':ref['boundary_error'],
        }
        assert physical>0 and alternate>0 and impedance.real>0
        assert row['oracle_reference_relative_error']<1e-8
        assert row['mobility_reference_relative_error']<1e-9
        assert row['dissipation_relative_error']<1e-8
        domain.append(row)

    refinement = []
    for radius,frequency,eta,temp in [(.5,.2,1.4,280.),(1.5,8.,.7,320.),
                                      (1.1,2.5,1.07,300.)]:
        coarse = reference.fluid_reference(radius*1e-6,frequency*1e6,eta*1e-3,temp)
        fine = reference.fluid_reference(radius*1e-6,frequency*1e6,eta*1e-3,temp,
                                          tolerance=1e-12,angular_order=32)
        refinement.append(abs(coarse['spectrum']/fine['spectrum']-1))
    assert max(refinement)<1e-9

    # DC calibration supplies a strictly monotone 1/eta response. Check the
    # complete objective as well, including dynamic in-phase and quadrature data.
    identification = []
    grid = np.linspace(.7,1.4,701)
    for eta in [.7,.71,.8,.95,TRUE,1.2,1.3,1.39,1.4]:
        y = oracle.predict_at(inputs,eta)
        data = [{'input':e,'value':float(v),'sigma':SIGMA} for e,v in zip(inputs,y)]
        fit = oracle.Model().fit(data).viscosity
        objective = np.array([np.mean((oracle.predict_at(inputs,x)-y)**2) for x in grid])
        local = int(np.sum((objective[1:-1]<objective[:-2])&(objective[1:-1]<objective[2:])))
        identification.append({'true':eta,'fit':fit,'relative_error':abs(fit/eta-1),
                               'interior_grid_minima':local})
        assert abs(fit/eta-1)<1e-7
        assert local==(0 if eta in [.7,1.4] else 1)

    equivalence = 0.
    for eta in [.7,.9,1.07,1.4]:
        equivalence = max(equivalence,float(np.max(abs(
            oracle.predict_at(inputs,eta)-shortcut.predict_at(inputs,eta)))))
    assert equivalence==0
    dc = reference.experiment(1.,0.,300.)
    assert oracle.predict_at([dc],TRUE)[0]==shortcut.predict_at([dc],TRUE)[0]
    even = conjugacy = fdt = ratio_error = 0.
    for radius,frequency,eta,temp in cases:
        e = reference.experiment(radius,frequency,temp)
        negative = dict(e,angular_frequency=-frequency)
        exact = oracle.predict_at([e],eta)[0]
        source = shortcut.predict_at([e],eta)[0]
        even = max(even,abs(oracle.predict_at([negative],eta)[0]/exact-1))
        mu = oracle.mobility(radius,frequency,eta)
        conjugacy = max(conjugacy,abs(oracle.mobility(radius,-frequency,eta)-mu.conjugate())/abs(mu))
        fdt = max(fdt,abs(exact/(2*oracle.KB*temp*mu.real*1e12)-1))
        ratio = 1+radius*1e-6*np.sqrt(frequency*1e6*1000/(2*eta*1e-3))
        ratio_error = max(ratio_error,abs(exact/source/ratio-1))
    assert max(even,conjugacy,fdt,ratio_error)<1e-12
    assert oracle.predict_at([],TRUE).shape==(0,)
    fits_all = [v[n] for v in fits for n in ['oracle','shortcut']]
    report = {
        'task':'thermal-unsteady-sphere','revision':1,'status':'scientific_validation_passed_no_model_evaluation',
        'nominal':nominal,
        'data_integrity':{'record_count':len(records),
            'distinct_calibration_settings':len({json.dumps(e,sort_keys=True) for e in inputs}),
            'repetitions':16,'fixed_sigma':SIGMA,'sigma_depends_on_response_or_unknown':False,
            'public_private_identical':True,'calibration_seed':SEED,'noise_seed':NOISE_SEED},
        'noise':{'draws':len(fits),'oracle_passes':256,'shortcut_rejections':256,
            'max_calibration_chi2':max(v['calibration_chi2'] for v in fits_all),
            'max_parameter_relative_error':max(v['parameter_relative_error'] for v in fits_all),
            'max_oracle_hidden_error':max(max(v['oracle']['hidden'].values()) for v in fits),
            'minimum_shortcut_diagnostic_error':min(v['shortcut']['hidden'][k] for v in fits
                for k in ['frequency_scan','radius_scan','temperature_controls'])},
        'domain':{'case_count':len(domain),'cases':domain,
            'max_reference_relative_error':max(d['oracle_reference_relative_error'] for d in domain),
            'max_mobility_reference_relative_error':max(d['mobility_reference_relative_error'] for d in domain),
            'max_dissipation_identity_relative_error':max(d['dissipation_relative_error'] for d in domain)},
        'refinement_relative_change_max':max(refinement),'identifiability':identification,
        'calibration_exact_equivalence_error':equivalence,
        'spectral_evenness_error':even,'mobility_conjugacy_error':conjugacy,
        'equilibrium_fdt_relative_error':fdt,'analytic_source_ratio_relative_error':ratio_error,
        'minimum_hidden_psd':min(float(np.min(truth[k])) for k in
            ['frequency_scan','radius_scan','temperature_controls']),
        'cold_hidden_reference_seconds':reference_seconds,
        'seconds':time.perf_counter()-start,
        'source_sha256':{str(p.relative_to(STAGE)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted([*TASK.rglob('*'),STAGE/'scripts/thermal_unsteady_sphere_baseline.py',Path(__file__)])
            if p.is_file() and '__pycache__' not in str(p)},
    }
    output = STAGE/'results/thermal-unsteady-sphere-r1-validation.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','domain']},indent=2))


if __name__=='__main__':
    main()
