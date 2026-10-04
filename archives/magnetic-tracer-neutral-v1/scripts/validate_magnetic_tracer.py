"""Validate magnetic-tracer controls, physical limits, and 256 noise realizations."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = 'magnetic-tracer'


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
    correct=module(ROOT/'tasks'/TASK/'solution/model.py')
    wrong=module(ROOT/'scripts/magnetic_tracer_baseline.py')
    reference_error=[];refinement=[];grid=[];minimum=[];dissipation=[];symmetry=[];mass=[]
    for cases in reference.hidden_inputs().values():
        expected=model.predict(cases)
        coarse=reference.predict(cases,true[0]);fine=reference.predict(cases,true[0],order=60)
        reference_error.append(float(np.max(abs(expected-coarse))))
        refinement.append(float(np.max(abs(fine-coarse))))
        e=cases[0];kx,ky=e['initial_wave']
        states=[]
        for points in [64,96]:
            x,rates,vectors,inverse=correct.modes(true[0],e['field_offset'],e['field_amplitude'],e['field_phase'],ky,points)
            z=inverse@np.exp(1j*(kx*x+e['initial_phase']))
            state=vectors@(np.exp(rates*e['time'])*z)
            states.append(e['amplitude']/2*np.real(np.mean(state*np.exp(-1j*(e['detector_mode']*x+e['detector_phase'])))))
        grid.append(abs(states[1]-states[0]))
        mirrored=[dict(r,field_offset=-r['field_offset'],field_amplitude=-r['field_amplitude'],
                       initial_wave=[-r['initial_wave'][0],r['initial_wave'][1]],initial_phase=-r['initial_phase'],
                       detector_mode=-r['detector_mode'],detector_phase=-r['detector_phase']) for r in cases]
        symmetry.append(float(np.max(abs(model.predict(mirrored)-model.predict(cases)))))
        for control in [correct,wrong]:
            x,rates,vectors,inverse=control.modes(true[0],e['field_offset'],e['field_amplitude'],e['field_phase'],ky)
            initial=np.exp(1j*(kx*x+e['initial_phase']));z=inverse@initial
            norms=[]
            for time in np.linspace(0,6,25):
                state=vectors@(np.exp(rates*time)*z)
                minimum.append(float(np.min(1-e['amplitude']*abs(state))))
                norms.append(float(np.mean(abs(state)**2)))
            dissipation.append(float(max(np.diff(norms))))
            x,rates,vectors,inverse=control.modes(true[0],e['field_offset'],e['field_amplitude'],e['field_phase'],0)
            equilibrium=vectors@(np.exp(rates*2)*(inverse@np.ones(len(x))))
            mass.append(float(np.max(abs(equilibrium-1))))
    assert max(reference_error)<2e-7 and max(refinement)<1e-9 and max(grid)<2e-7
    assert min(minimum)>.29 and max(dissipation)<1e-10 and max(symmetry)<1e-9 and max(mass)<1e-9
    return dict(independent_galerkin_max_error=max(reference_error),
                reference_40_vs_60_max_difference=max(refinement),collocation_64_vs_96_max_difference=max(grid),
                minimum_density_relative_to_uniform=min(minimum),maximum_L2_norm_increase=max(dissipation),
                field_reversal_reflection_max_error=max(symmetry),uniform_equilibrium_max_error=max(mass),
                both_controls_probability_conserving=True,exact_uniform_field_calibration=True)


if __name__ == '__main__':
    main()
