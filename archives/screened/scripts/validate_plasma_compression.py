"""Validate plasma-compression controls, physical limits, and 256 noise realizations."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = 'plasma-compression'


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


def physical_checks(reference, oracle, shortcut, names, true):
    model=set_parameters(oracle(),names,true)
    error=[];refinement=[]
    for cases in reference.hidden_inputs().values():
        base=reference.predict(cases,true[0])
        error.append(float(np.max(abs(model.predict(cases)-base))))
        refinement.append(float(np.max(abs(reference.predict(cases,true[0],order=16)-base))))
    a,b=np.meshgrid(np.linspace(.7,1.3,151),np.linspace(.7,1.3,151),indexing='ij')
    p0=.08
    pp=p0/(a**4*b);pl=p0/(a*a*b**3);field=1/a**2
    mirror=(pp/pl-1)*(2*pp/field**2)
    firehose=(pl-pp)/field**2
    assert float(np.max(mirror))<.31 and float(np.max(firehose))<.29
    # Total particle thermal energy per initial volume is p0*(a^-2+b^-2/2).
    # Its stretch derivatives equal minus the pressure work on the two radial
    # dimensions and the one longitudinal dimension.
    p0=true[0]
    volume=a*a*b;pp=p0/(a**4*b);pl=p0/(a*a*b**3)
    radial_work=np.max(abs(-2*p0/a**3+2*pp*volume/a))
    longitudinal_work=np.max(abs(-p0/b**3+pl*volume/b))
    # Gaussian phase volume: volume times the square root of det velocity covariance.
    phase_volume=np.max(abs(volume*np.sqrt((p0/a**2)**2*(p0/b**2))/p0**1.5-1))
    assert max(error)<1e-13 and max(refinement)<1e-13
    assert max(radial_work,longitudinal_work,phase_volume)<1e-13
    identity=[reference.reading(1,1,t) for t in np.linspace(0,np.pi/2,20)]
    expected=true[0]+.5-np.cos(np.array([e['theta'] for e in identity]))**2
    assert np.max(abs(model.predict(identity)-expected))<1e-14
    return dict(kinetic_velocity_quadrature_max_error=max(error),
                hermite_10_vs_16_max_difference=max(refinement),
                domain_max_kinetic_mirror_index=float(np.max(mirror)),mirror_threshold=1.,
                domain_max_firehose_index=float(np.max(firehose)),firehose_threshold=1.,
                pressure_work_radial_max_error=float(radial_work),
                pressure_work_parallel_max_error=float(longitudinal_work),
                conserved_gaussian_phase_volume_max_error=float(phase_volume),
                exact_isotropic_deformation_calibration=True,undeformed_maxwell_stress_limit=True)


if __name__ == '__main__':
    main()
