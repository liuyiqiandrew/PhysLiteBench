"""Validate compensated-conductor controls, physical limits, and 256 noise realizations."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = 'compensated-conductor'


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
    sigma = metadata['measurement_sigma_fraction']*np.max(abs(noiseless))
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
        assert delta < 1e-9
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


def physical_checks(reference, oracle, shortcut, names, true):
    model = set_parameters(oracle(), names, true)
    baseline = set_parameters(shortcut(), names, true)
    mu = true[0]
    charge = reference.CHARGE
    rng = np.random.default_rng(4502)
    reference_error = []
    transverse_error = []
    dissipation_error = []
    symmetry_error = []
    for _ in range(100):
        e = dict(electric_field=float(rng.uniform(-3., 3.)),
                 magnetic_field=float(rng.uniform(-6., 6.)),
                 density_positive=float(rng.uniform(.1e21, 4e21)),
                 density_negative=float(rng.uniform(.1e21, 4e21)))
        result = model.predict([e])[0]
        state = reference.state(e, mu)
        np_, nm = e['density_positive'], e['density_negative']
        current = charge*np.array([np_*state[0]-nm*state[2], np_*state[1]-nm*state[3]])
        reference_error.append(abs(result-current[0]))
        transverse_error.append(abs(current[1]))
        power = float(current@[e['electric_field'], state[4]])
        friction = charge/mu*(np_*sum(state[:2]**2)+nm*sum(state[2:4]**2))
        dissipation_error.append(abs(power-friction)/max(abs(power), 1e-30))
        assert power >= 0
        reversed_field = dict(e, magnetic_field=-e['magnetic_field'])
        reversed_drive = dict(e, electric_field=-e['electric_field'])
        swapped = dict(e, density_positive=nm, density_negative=np_)
        doubled = dict(e, density_positive=2*np_, density_negative=2*nm)
        symmetry_error.extend([abs(model.predict([reversed_field])[0]-result),
                               abs(model.predict([reversed_drive])[0]+result),
                               abs(model.predict([swapped])[0]-result),
                               abs(reference.state(reversed_field, mu)[4]+state[4]),
                               abs(reference.state(swapped, mu)[4]+state[4]),
                               abs(model.predict([doubled])[0]-2*result)])
        zero_field = dict(e, magnetic_field=0.)
        expected = charge*(np_+nm)*mu*e['electric_field']
        assert abs(model.predict([zero_field])[0]-expected) < 1e-10
        equal = dict(e, density_negative=np_)
        assert abs(model.predict([equal])[0]-baseline.predict([equal])[0]) < 1e-10
        assert abs(reference.state(equal, mu)[4]) < 1e-12
        for single in [dict(e, density_positive=0.), dict(e, density_negative=0.)]:
            expected = charge*(single['density_positive']+single['density_negative'])*mu*e['electric_field']
            assert abs(model.predict([single])[0]-expected) < 1e-10
            assert abs(reference.predict([single], mu)[0]-expected) < 1e-10
        assert model.predict([dict(e, electric_field=0.)])[0] == 0
    inputs = reference.calibration_inputs()
    truth = reference.predict(inputs, mu)
    grid = np.linspace(.08, 1.4, 1001)
    loss = []
    for value in grid:
        model.mobility = value
        loss.append(float(np.sum((model.predict(inputs)-truth)**2)))
    index = int(np.argmin(loss))
    assert abs(grid[index]-mu) < grid[1]-grid[0]
    assert np.all(np.diff(loss[:index+1]) < 0) and np.all(np.diff(loss[index:]) > 0)
    assert max(reference_error) < 1e-10 and max(transverse_error) < 1e-10
    assert max(dissipation_error) < 1e-12 and max(symmetry_error) < 1e-10
    return dict(independent_reference_max_error=float(max(reference_error)),
                total_transverse_current_max=float(max(transverse_error)),
                dissipation_relative_error_max=float(max(dissipation_error)),
                symmetry_max_error=float(max(symmetry_error)),
                calibration_loss_single_minimum_on_1001_point_grid=True,
                zero_field_limit=True, single_carrier_limit=True,
                compensation_limit=True, zero_drive_limit=True)


if __name__ == '__main__':
    main()
