"""Validate spatial-diffusion revision 1; regenerate calibration only with --generate."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['spatial-diffusion']


def module(path):
    spec = importlib.util.spec_from_file_location('candidate_'+str(abs(hash(str(path)))), path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def scale_for(name, experiments, truth):
    if name == 'spatial-diffusion':
        return np.ones_like(truth)
    return np.full_like(truth, np.sqrt(np.mean(truth**2)))


def calibration_metrics(model, records, reference):
    values = np.array([r['value'] for r in records])
    sigma = np.array([r['sigma'] for r in records])
    pred = model.predict([r['input'] for r in records])
    return dict(parameter=float(getattr(model, reference.PARAMETER)),
                parameter_relative_error=float(abs(getattr(model, reference.PARAMETER)/reference.TRUE_PARAMETER-1)),
                calibration_chi2=float(np.sum(((pred-values)/sigma)**2)/(len(records)-1)))


def validate(name, generate, noise_trials):
    task = ROOT/'tasks'/name
    ref = module(task/'tests/reference.py')
    oracle_class = module(task/'solution/model.py').Model
    baseline_class = module(ROOT/'scripts'/(name.replace('-', '_')+'_baseline.py')).Model
    meta = json.loads((task/'tests/metadata.json').read_text())
    inputs = ref.calibration_inputs()*2
    noiseless = ref.predict(inputs, ref.TRUE_PARAMETER)
    sigma = .006*max(np.max(np.abs(noiseless)), 1e-10)
    seed = 9210
    if generate:
        noise = np.random.default_rng(seed).normal(0, sigma, len(inputs))
        records = [dict(input=e, value=float(v), sigma=float(sigma)) for e, v in zip(inputs, noiseless+noise)]
        for folder in ['environment', 'tests']:
            path = task/folder/'data/calibration.json'
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(records, indent=2)+'\n')
    records = json.loads((task/'environment/data/calibration.json').read_text())
    assert records == json.loads((task/'tests/data/calibration.json').read_text())
    report = {'revision':1, 'seed':seed, 'noise_trials':noise_trials, 'controls':{}, 'noise':{}}
    hidden = ref.hidden_inputs()
    truths = {key: ref.predict(es, ref.TRUE_PARAMETER) for key, es in hidden.items()}
    for label, cls in [('oracle', oracle_class), ('shortcut', baseline_class)]:
        exact = cls()
        setattr(exact, ref.PARAMETER, ref.TRUE_PARAMETER)
        calibration_delta = float(np.max(np.abs(exact.predict(inputs)-noiseless)))
        assert calibration_delta < 1e-6*max(1., np.max(np.abs(noiseless)))
        model = cls().fit(records)
        metrics = calibration_metrics(model, records, ref)
        assert metrics['calibration_chi2'] < 1.5 and metrics['parameter_relative_error'] < .03
        metrics['calibration_reference_max_error'] = calibration_delta
        metrics['hidden'] = {}
        for key, experiments in hidden.items():
            truth = truths[key]
            score = float(np.sqrt(np.mean(((model.predict(experiments)-truth)/scale_for(name, experiments, truth))**2)))
            metrics['hidden'][key] = score
        if label == 'oracle':
            assert max(metrics['hidden'].values()) < meta['prediction_limit'], metrics
        else:
            assert min(metrics['hidden'].values()) > meta['prediction_limit'], metrics
        report['controls'][label] = metrics

    # Both controls have exactly the same calibration predictor. Fit each noise
    # realization once, then verify the other fit on the first and last cases.
    rng = np.random.default_rng(seed+10000)
    fitted, chi2s = [], []
    for k in range(noise_trials):
        noisy = noiseless+rng.normal(0, sigma, len(inputs))
        sample = [dict(input=e, value=float(v), sigma=float(sigma)) for e, v in zip(inputs, noisy)]
        model = oracle_class().fit(sample)
        metrics = calibration_metrics(model, sample, ref)
        fitted.append(metrics['parameter'])
        chi2s.append(metrics['calibration_chi2'])
        assert metrics['parameter_relative_error'] < .03, metrics
        if k in [0, noise_trials-1]:
            other = baseline_class().fit(sample)
            assert abs(getattr(other, ref.PARAMETER)/metrics['parameter']-1) < 1e-5
    report['noise'] = {'chi2_max':max(chi2s), 'chi2_pass_fraction':float(np.mean(np.array(chi2s)<1.5)),
                       'parameter_min':min(fitted), 'parameter_max':max(fitted),
                       'parameter_relative_error_max':float(np.max(np.abs(np.array(fitted)/ref.TRUE_PARAMETER-1)))}
    assert report['noise']['chi2_pass_fraction'] >= .99
    # Check hidden predictions at the observed fitted-parameter extrema. This is
    # explicitly a sensitivity check, not a claim of exhaustive hidden MC trials.
    sensitivity = {}
    for label, cls in [('oracle', oracle_class), ('shortcut', baseline_class)]:
        scores = []
        for p in [min(fitted), max(fitted)]:
            model = cls()
            setattr(model, ref.PARAMETER, p)
            for key, es in hidden.items():
                score = float(np.sqrt(np.mean(((model.predict(es)-truths[key])/scale_for(name, es, truths[key]))**2)))
                scores.append(score)
        sensitivity[label] = {'min':min(scores), 'max':max(scores)}
    assert sensitivity['oracle']['max'] < meta['prediction_limit']
    assert sensitivity['shortcut']['min'] > meta['prediction_limit']
    report['hidden_parameter_extrema_check'] = sensitivity

    report['independent_checks'] = independent_checks(ref, oracle_class, baseline_class)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tasks', nargs='*', choices=NAMES)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--noise-trials', type=int, default=256)
    parser.add_argument('--output', type=Path, default=ROOT/'jobs/spatial-diffusion-validation/summary.json')
    args = parser.parse_args()
    if args.noise_trials < 1:
        parser.error('noise-trials must be positive')
    output = {}
    for name in args.tasks or NAMES:
        start = time.monotonic()
        output[name] = validate(name, args.generate, args.noise_trials)
        output[name]['seconds'] = time.monotonic()-start
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2)+'\n')
        print(name, 'PASS', round(output[name]['seconds'], 2), flush=True)


def independent_checks(ref, oracle_class, baseline_class):
    from scipy.integrate import solve_ivp
    from scipy.sparse import diags
    # Convergence is checked across every private prediction, including an
    # initially uniform ensemble and the largest allowed drag contrast.
    errors = {}
    all_es = [e for es in ref.hidden_inputs().values() for e in es]
    for cells in [256, 512, 1024]:
        truth = ref.predict(all_es, ref.TRUE_PARAMETER, cells=cells)
        oracle = oracle_class()
        oracle.diffusivity = ref.TRUE_PARAMETER
        errors[str(cells)] = float(np.max(abs(truth-oracle.predict(all_es))))
    assert errors['512'] < 3e-5 and errors['1024'] < errors['512']/3
    matrix = ref.flux_operator(.8, ref.TRUE_PARAMETER, 512)
    assert np.max(abs(np.asarray(matrix.sum(axis=0)))) < 1e-10
    assert np.max(abs(matrix@np.ones(512))) < 1e-10
    trajectory = ref.trajectory(.7, .5, -.15, ref.TRUE_PARAMETER, 80., 512)
    probabilities = trajectory.sol(np.linspace(0., 80., 50))
    mass_error = float(np.max(abs(probabilities.sum(axis=0)-1)))
    assert mass_error < 1e-10 and probabilities.min() >= 0

    # Independently verify that the shortcut really solves zero-drift Ito
    # diffusion, rather than failing due to a faulty Fourier implementation.
    cells = 1024
    theta = 2*np.pi*(np.arange(cells)+.5)/cells
    a, first, second = .8, .5, -.2
    diffusivity = ref.TRUE_PARAMETER
    laplace = ref.flux_operator(0., 1., cells)
    wrong_matrix = laplace@diags(diffusivity*(1+a*np.cos(theta)))
    initial = (1+first*np.cos(theta)+second*np.cos(2*theta))/cells
    answer = solve_ivp(lambda t, p: wrong_matrix@p, (0., 65.), initial,
                       method='BDF', jac=wrong_matrix, rtol=2e-10, atol=1e-12, dense_output=True)
    assert answer.success
    es = [dict(contrast=a, first=first, second=second, time=float(t), mode=m)
          for m in [1, 2] for t in np.geomspace(.2, 65., 25)]
    finite_volume = np.array([np.cos(e['mode']*theta)@answer.sol(e['time']) for e in es])
    baseline = baseline_class()
    baseline.diffusivity = diffusivity
    shortcut_error = float(np.max(abs(finite_volume-baseline.predict(es))))
    assert shortcut_error < 3e-5
    oracle = oracle_class()
    oracle.diffusivity = diffusivity
    uniform = [dict(contrast=a, first=0., second=0., time=80., mode=m) for m in [1, 2]]
    uniform_error = float(np.max(abs(oracle.predict(uniform))))
    assert uniform_error < 1e-13
    return {'finite_volume_error_by_cells': errors, 'mass_error': mass_error,
            'minimum_cell_probability': float(probabilities.min()),
            'shortcut_independent_ito_error': shortcut_error,
            'uniform_equilibrium_error': uniform_error}



if __name__ == '__main__':
    main()
