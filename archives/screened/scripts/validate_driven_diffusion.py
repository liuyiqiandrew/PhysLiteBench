"""Validate driven-diffusion controls, physical limits, and 256 noise realizations."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = 'driven-diffusion'


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
    physical=module(ROOT/'tasks'/TASK/'solution/model.py')
    baseline=module(ROOT/'scripts/driven_diffusion_baseline.py')
    errors=[];refinements=[];quadrature=[];derivatives=[];stationarity=[];correctors=[];positive=[];drift_equal=[]
    for A in [0.,.5,1.,2.,3.]:
        for F in [-3.,-1.2,0.,.7,2.,3.]:
            v,d=physical.transport(A,F)
            coarse=reference.cell(A,F);fine=reference.cell(A,F,points=128)
            errors.append(max(abs(v-coarse[0]),abs(d-coarse[1])))
            refinements.append(max(abs(fine[0]-coarse[0]),abs(fine[1]-coarse[1])))
            q=physical.transport(A,F,points=256,quadrature=128)
            quadrature.append(max(abs(q[0]-v),abs(q[1]-d)))
            vb,mobility=baseline.transport(A,F)
            drift_equal.append(abs(vb-v))
            step=1e-4
            derivative=(baseline.transport(A,F-2*step)[0]-8*baseline.transport(A,F-step)[0]
                        +8*baseline.transport(A,F+step)[0]-baseline.transport(A,F+2*step)[0])/(12*step)
            derivatives.append(abs(derivative-mobility))
            probability,chi,L=coarse[2:]
            x,first,second=reference.derivatives(96)
            local=F+A*np.sin(x)
            stationarity.append(float(np.max(abs(L.T@probability))))
            correctors.append(float(np.max(abs(L@chi-(coarse[0]-local)))))
            positive.append(float(probability.min()))
            if F==0:
                from scipy.special import i0
                equilibrium=1/i0(A)**2
                assert abs(d-equilibrium)<1e-12 and abs(mobility-equilibrium)<1e-12
            if A==0:
                assert abs(v-F)<1e-12 and abs(d-1)<1e-12 and abs(mobility-1)<1e-12
            reverse=physical.transport(A,-F)
            assert abs(reverse[0]+v)<1e-12 and abs(reverse[1]-d)<1e-12
    assert max(errors)<1e-9 and max(refinements)<1e-9 and max(quadrature)<1e-11
    assert max(derivatives)<1e-8 and max(stationarity)<1e-10 and max(correctors)<1e-9
    assert min(positive)>0 and max(drift_equal)<1e-14
    return dict(independent_cell_problem_max_error=float(max(errors)),
                reference_96_vs_128_max_difference=float(max(refinements)),
                quadrature_refinement_max_difference=float(max(quadrature)),
                analytical_differential_mobility_vs_five_point_derivative_max_error=float(max(derivatives)),
                stationary_probability_residual_max=float(max(stationarity)),
                displacement_corrector_residual_max=float(max(correctors)),
                stationary_probability_min=float(min(positive)),
                controls_exact_drift_agreement=True,flat_potential_limit=True,
                zero_force_lifson_jackson_and_einstein_limits=True,force_reversal_symmetry=True)


if __name__ == '__main__':
    main()
