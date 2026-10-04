"""Validate thermal-transition-basis controls, physical limits, and 256 noise realizations."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = 'thermal-transition-basis'


def module(path):
    spec = importlib.util.spec_from_file_location('validation_'+str(abs(hash(str(path)))), path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def set_parameters(model, names, values):
    for name, value in zip(names, values):
        setattr(model, name, float(value))
    return model


def metrics(model, records, metadata):
    y = model.predict([r['input'] for r in records])
    values = np.array([r['value'] for r in records])
    sigma = np.array([r['sigma'] for r in records])
    parameters = np.array([getattr(model, name) for name in metadata['parameters']])
    return dict(parameters=parameters.tolist(),
                parameter_relative_error_max=float(np.max(abs(parameters/np.array(metadata['true_parameters'])-1))),
                calibration_chi2=float(np.sum(((y-values)/sigma)**2)/(len(records)-len(parameters))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--noise-trials', type=int, default=256)
    parser.add_argument('--output', type=Path, default=ROOT/'jobs'/(TASK+'-validation')/'summary.json')
    args = parser.parse_args()
    if args.noise_trials < 1:
        parser.error('noise-trials must be positive')
    task = ROOT/'tasks'/TASK
    reference = module(task/'tests/reference.py')
    oracle = module(task/'solution/model.py').Model
    shortcut = module(ROOT/'scripts'/(TASK.replace('-', '_')+'_baseline.py')).Model
    metadata = json.loads((task/'tests/metadata.json').read_text())
    names, true = metadata['parameters'], metadata['true_parameters']
    argument = true[0] if len(true) == 1 else true
    inputs = reference.calibration_inputs()*2
    noiseless = reference.predict(inputs, argument)
    sigma = metadata['measurement_sigma']
    if args.generate:
        values = noiseless+np.random.default_rng(metadata['calibration_seed']).normal(0, sigma, len(inputs))
        records = [dict(input=e, value=float(value), sigma=float(sigma)) for e, value in zip(inputs, values)]
        for folder in ['environment', 'tests']:
            (task/folder/'data/calibration.json').write_text(json.dumps(records, indent=2)+'\n')
    records = json.loads((task/'environment/data/calibration.json').read_text())
    assert records == json.loads((task/'tests/data/calibration.json').read_text())
    assert [r['input'] for r in records] == inputs
    hidden = reference.hidden_inputs()
    truths = {name: reference.predict(experiments, argument) for name, experiments in hidden.items()}
    def hidden_scores(model):
        return {name: float(np.linalg.norm(model.predict(hidden[name])-truth)/np.linalg.norm(truth))
                for name, truth in truths.items()}
    report = dict(revision=metadata['revision'], calibration_seed=metadata['calibration_seed'],
                  noise_seed=metadata['noise_seed'], noise_trials=args.noise_trials,
                  measurement_sigma=float(sigma), controls={})
    for label, cls in [('oracle', oracle), ('shortcut', shortcut)]:
        exact = set_parameters(cls(), names, true)
        delta = float(np.max(abs(exact.predict(inputs)-noiseless)))
        assert delta < 3e-8
        model = cls().fit(records)
        result = metrics(model, records, metadata)
        result['calibration_reference_max_error'] = delta
        result['hidden'] = hidden_scores(model)
        assert result['parameter_relative_error_max'] < .03 and result['calibration_chi2'] < 1.5
        if label == 'oracle':
            assert max(result['hidden'].values()) < metadata['prediction_limit']
        else:
            assert min(result['hidden'].values()) > metadata['prediction_limit']
        report['controls'][label] = result
    report['physical_checks'] = physical_checks(reference, oracle, shortcut, names, true)
    rng = np.random.default_rng(metadata['noise_seed'])
    parameters, chi2s, errors = [], [], []
    noise_hidden = {'oracle': [], 'shortcut': []}
    for trial in range(args.noise_trials):
        values = noiseless+rng.normal(0, sigma, len(inputs))
        sample = [dict(input=e, value=float(value), sigma=float(sigma)) for e, value in zip(inputs, values)]
        oracle_model = oracle().fit(sample)
        shortcut_model = shortcut().fit(sample)
        result = metrics(oracle_model, sample, metadata)
        parameters.append(result['parameters'])
        chi2s.append(result['calibration_chi2'])
        errors.append(result['parameter_relative_error_max'])
        assert result['parameter_relative_error_max'] < .03
        assert np.max(abs(np.array(result['parameters'])-np.array([getattr(shortcut_model, n) for n in names]))) < 1e-6
        for label, model in [('oracle', oracle_model), ('shortcut', shortcut_model)]:
            noise_hidden[label].extend(hidden_scores(model).values())
    report['noise'] = dict(parameter_min=np.min(parameters, axis=0).tolist(),
                          parameter_max=np.max(parameters, axis=0).tolist(),
                          parameter_relative_error_max=max(errors),
                          calibration_chi2_max=max(chi2s),
                          calibration_pass_fraction=float(np.mean(np.array(chi2s)<1.5)),
                          hidden={label: dict(min=min(scores), max=max(scores)) for label, scores in noise_hidden.items()})
    assert report['noise']['calibration_pass_fraction'] >= .99
    assert max(noise_hidden['oracle']) < metadata['prediction_limit']
    assert min(noise_hidden['shortcut']) > metadata['prediction_limit']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(TASK, 'PASS', args.output)


def physical_checks(reference,oracle,shortcut,names,true):
    physical=module(ROOT/'tasks'/TASK/'solution/model.py')
    baseline=module(ROOT/'scripts/thermal_transition_basis_baseline.py')
    correct=set_parameters(oracle(),names,true)
    errors=[];trace=[];balance=[];stationarity=[];local_failure=[];positivity=[];currents=[]
    for cases in reference.hidden_inputs().values():
        errors.append(float(np.max(abs(correct.predict(cases)-reference.predict(cases,true[0])))))
    for coupling in [0.,.25,.45,.65,.7]:
        h,energies,vectors,reservoirs=physical.setup(coupling)
        population,observables,gibbs,states,values=reference.population_model(coupling)
        rho=(vectors*gibbs)@vectors.conj().T
        balance.append(float(np.max(abs(population*gibbs[None,:]-(population*gibbs[None,:]).T))))
        for code in [physical,baseline]:
            basis,generator,rates,modes,inverse,steady=code.dynamics(coupling,true[0])
            trace.append(float(np.max(abs(np.eye(4).ravel(order='F')@generator))))
            positivity.append(float(np.linalg.eigvalsh((steady+steady.conj().T)/2).min()))
            assert np.max(abs(generator@steady.ravel(order='F')))<1e-11
            if code is physical:stationarity.append(float(np.max(abs(steady-rho))))
            elif coupling>0:
                local_failure.append(float(np.linalg.norm(steady-rho)))
                bath=code.setup(coupling)[3]
                currents.append(float(abs(np.trace(h@(true[0]*bath[0]@steady.ravel(order='F')).reshape(4,4,order='F')).real)))
        for bath in reservoirs:
            assert np.max(abs(bath@rho.ravel(order='F')))<1e-11
    assert max(errors)<1e-9 and max(trace)<1e-12 and max(balance)<1e-12
    assert max(stationarity)<1e-11 and min(positivity)>0 and min(local_failure)>.01
    assert min(currents)>1e-6
    return dict(independent_population_reference_max_error=max(errors),
                full_hamiltonian_detailed_balance_max_error=max(balance),
                both_controls_trace_preservation_max_error=max(trace),
                correct_steady_gibbs_max_error=max(stationarity),
                both_steady_states_minimum_eigenvalue=min(positivity),
                shortcut_stationary_gibbs_distance_min=min(local_failure),
                shortcut_same_temperature_heat_current_min=min(currents),
                zero_coupling_calibration_limit=True,
                each_correct_bath_annihilates_gibbs=True)


if __name__ == '__main__':
    main()
