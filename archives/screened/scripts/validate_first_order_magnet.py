"""Validate first-order-magnet controls, branch physics, and calibration noise."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
TASK = 'first-order-magnet'


def module(path):
    spec = importlib.util.spec_from_file_location('validation_'+str(abs(hash(str(path)))), path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def metrics(model, records, true):
    y = model.predict([r['input'] for r in records])
    residual = (y-np.array([r['value'] for r in records]))/np.array([r['sigma'] for r in records])
    return dict(parameter=model.coupling, parameter_relative_error=abs(model.coupling/true-1),
                calibration_chi2=float(residual@residual)/(len(records)-1))


def branch_checks(oracle_module, baseline, experiments):
    """Check the shortcut remains a physical metastable phase, not an invalid root."""
    coupling, q = baseline.coupling, oracle_module.QUARTIC_COUPLING
    small = baseline.predict(experiments)
    ordered = oracle_module.Model()
    ordered.coupling = coupling
    equilibrium = ordered.predict(experiments)
    residuals, curvatures, gaps, competitor_gaps = [], [], [], []
    grid = np.linspace(-1., 1., 4001)
    for e, m, best in zip(experiments, small, equilibrium):
        t, h = e['temperature'], e['field']
        equation = lambda x: x-np.tanh((coupling*x+q*x**3+h)/t)
        curvature = lambda x: t/(1-x*x)-coupling-3*q*x*x
        energy = lambda x: oracle_module.free_energy(x, t, h, coupling)
        residuals.append(abs(equation(m)))
        curvatures.append(curvature(m))
        gaps.append(energy(m)-energy(best))
        assert m*h > 0 and best*h > 0 and abs(m) < 1 and abs(best) < 1
        assert curvature(best) > 0
        values = equation(grid)
        roots = [brentq(equation, grid[i], grid[i+1], xtol=1e-13)
                 for i in np.flatnonzero(values[:-1]*values[1:] < 0)]
        competing = [r for r in roots if curvature(r) > 0 and abs(r-best) > 1e-6]
        assert competing
        competitor_gaps.extend(energy(r)-energy(best) for r in competing)
    result = dict(shortcut_residual_max=float(max(residuals)),
                  shortcut_curvature_min=float(min(curvatures)),
                  shortcut_free_energy_gap_min=float(min(gaps)),
                  all_competing_minima_free_energy_gap_min=float(min(competitor_gaps)))
    assert result['shortcut_residual_max'] < 1e-10
    assert result['shortcut_curvature_min'] > .05
    assert result['shortcut_free_energy_gap_min'] > .005
    assert result['all_competing_minima_free_energy_gap_min'] > .004
    return result


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
    oracle_module = module(task/'solution/model.py')
    oracle = oracle_module.Model
    shortcut = module(ROOT/'scripts/first_order_magnet_baseline.py').Model
    metadata = json.loads((task/'tests/metadata.json').read_text())
    true = metadata['true_parameter']
    inputs = reference.calibration_inputs()*2
    noiseless = reference.predict(inputs, true)
    sigma = metadata['measurement_sigma_fraction']*np.max(abs(noiseless))
    if args.generate:
        values = noiseless+np.random.default_rng(metadata['calibration_seed']).normal(0, sigma, len(inputs))
        records = [dict(input=e, value=float(value), sigma=float(sigma)) for e, value in zip(inputs, values)]
        for folder in ['environment', 'tests']:
            (task/folder/'data/calibration.json').write_text(json.dumps(records, indent=2)+'\n')
    public, private = task/'environment/data/calibration.json', task/'tests/data/calibration.json'
    assert public.read_bytes() == private.read_bytes()
    records = json.loads(public.read_text())
    assert [r['input'] for r in records] == inputs
    hidden = reference.hidden_inputs()
    experiments = [e for group in hidden.values() for e in group]
    truths = {name: reference.predict(es, true) for name, es in hidden.items()}
    def scores(model):
        return {name: float(np.sqrt(np.mean((model.predict(hidden[name])-truth)**2)))
                for name, truth in truths.items()}
    report = dict(revision=metadata['revision'], calibration_seed=metadata['calibration_seed'],
                  noise_seed=metadata['noise_seed'], noise_trials=args.noise_trials,
                  measurement_sigma=float(sigma), controls={})
    for label, cls in [('oracle', oracle), ('shortcut', shortcut)]:
        exact = cls()
        exact.coupling = true
        delta = float(np.max(abs(exact.predict(inputs)-noiseless)))
        assert delta < 1e-7
        model = cls().fit(records)
        result = metrics(model, records, true)
        result['calibration_reference_max_error'] = delta
        result['hidden'] = scores(model)
        assert result['parameter_relative_error'] < .03 and result['calibration_chi2'] < 1.5
        assert (max(result['hidden'].values()) < metadata['prediction_limit'] if label == 'oracle'
                else min(result['hidden'].values()) > metadata['prediction_limit'])
        report['controls'][label] = result
    exact = oracle()
    exact.coupling = true
    truth = reference.predict(experiments, true)
    reference_error = float(np.max(abs(exact.predict(experiments)-truth)))
    assert reference_error < 1e-7
    reversed_fields = [dict(e, field=-e['field']) for e in experiments]
    symmetry_error = float(np.max(abs(exact.predict(experiments)+exact.predict(reversed_fields))))
    assert symmetry_error < 1e-10
    min_t = min(e['temperature'] for e in inputs)
    q = oracle_module.QUARTIC_COUPLING
    min_curvature = 2*np.sqrt(3*q*min_t)-3*q-metadata['bounds'][1]
    assert min_curvature > 0
    baseline = shortcut()
    baseline.coupling = true
    report['physical_checks'] = dict(independent_reference_max_error=reference_error,
                                     field_reversal_max_error=symmetry_error,
                                     calibration_curvature_lower_bound=float(min_curvature),
                                     **branch_checks(oracle_module, baseline, experiments))
    rng = np.random.default_rng(metadata['noise_seed'])
    parameters, chi2s, errors, branches = [], [], [], []
    noise_hidden = {'oracle': [], 'shortcut': []}
    for _ in range(args.noise_trials):
        values = noiseless+rng.normal(0, sigma, len(inputs))
        sample = [dict(input=e, value=float(value), sigma=float(sigma)) for e, value in zip(inputs, values)]
        models = {'oracle': oracle().fit(sample), 'shortcut': shortcut().fit(sample)}
        result = metrics(models['oracle'], sample, true)
        parameters.append(result['parameter'])
        chi2s.append(result['calibration_chi2'])
        errors.append(result['parameter_relative_error'])
        assert result['parameter_relative_error'] < .03
        assert abs(models['oracle'].coupling-models['shortcut'].coupling) < 1e-6
        for label, model in models.items():
            noise_hidden[label].extend(scores(model).values())
        branches.append(branch_checks(oracle_module, models['shortcut'], experiments))
    report['noise'] = dict(parameter_min=min(parameters), parameter_max=max(parameters),
                          parameter_relative_error_max=max(errors), calibration_chi2_max=max(chi2s),
                          calibration_pass_fraction=float(np.mean(np.array(chi2s)<1.5)),
                          hidden={label: dict(min=min(values), max=max(values)) for label, values in noise_hidden.items()},
                          branches={key: (max if key.endswith('_max') else min)(r[key] for r in branches)
                                    for key in branches[0]})
    assert report['noise']['calibration_pass_fraction'] >= .99
    assert max(noise_hidden['oracle']) < metadata['prediction_limit']
    assert min(noise_hidden['shortcut']) > metadata['prediction_limit']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(TASK, 'PASS', args.output)


if __name__ == '__main__':
    main()
