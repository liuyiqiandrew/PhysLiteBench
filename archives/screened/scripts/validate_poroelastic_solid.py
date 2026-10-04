"""Validate poroelastic-solid controls, physical limits, and 256 noise realizations."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = 'poroelastic-solid'


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
    components = reference.COMPONENTS
    def tensor(vector):
        xx, yy, zz, xy, xz, yz = vector
        return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    def experiments(strain):
        return [dict(strain=list(strain), component=c) for c in components]
    zero = model.predict(experiments([0, 0, 0, 0, 0, 0]))
    assert np.max(abs(zero)) == 0
    shear_experiments = experiments([.001, -.001, 0., .0004, -.0002, .0001])
    assert np.max(abs(model.predict(shear_experiments)-baseline.predict(shear_experiments))) < 1e-12
    bulk = true[0]/(3*(1-2*.25))+.8**2*2400.
    pressure_test = model.predict(experiments([-.001, -.001, -.001, 0., 0., 0.]))
    assert np.max(abs(pressure_test[:3]+.003*bulk)) < 1e-12
    assert np.max(abs(pressure_test[3:])) == 0
    rng = np.random.default_rng(4501)
    rotation_error = []
    fluid_error = []
    reference_error = []
    energy_extra = []
    for _ in range(40):
        strain = rng.uniform(-.0005, .0005, 6)
        stress = model.predict(experiments(strain))
        state = reference.state(strain, true[0])
        reference_error.append(float(np.max(abs(stress-state[:6]))))
        fluid_error.append(abs(.8*sum(strain[:3])+state[6]/2400.))
        assert np.max(abs(model.predict(experiments(-strain))+stress)) < 1e-12
        rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        rotated = rotation@tensor(strain)@rotation.T
        vector = [rotated[0, 0], rotated[1, 1], rotated[2, 2], rotated[0, 1], rotated[0, 2], rotated[1, 2]]
        rotation_error.append(float(np.max(abs(tensor(model.predict(experiments(vector)))-rotation@tensor(stress)@rotation.T))))
        stress_extra = tensor(stress-baseline.predict(experiments(strain)))
        extra = .5*float(np.sum(stress_extra*tensor(strain)))
        expected = .5*2400.*.8**2*sum(strain[:3])**2
        assert abs(extra-expected) < 1e-14
        energy_extra.append(extra)
        uncoupled = reference.state(strain, true[0], alpha=0.)[:6]
        assert np.max(abs(uncoupled-baseline.predict(experiments(strain)))) < 1e-12
    assert max(rotation_error) < 1e-12 and max(fluid_error) < 1e-12
    assert max(reference_error) < 1e-12 and min(energy_extra) >= 0
    return dict(undrained_bulk_modulus=bulk, independent_reference_max_error=max(reference_error),
                rotation_max_error=max(rotation_error), fluid_content_max_error=float(max(fluid_error)),
                minimum_added_fluid_energy=min(energy_extra), zero_strain=True,
                shear_limit=True, zero_biot_coupling=True, hydrostatic_limit=True)


if __name__ == '__main__':
    main()
