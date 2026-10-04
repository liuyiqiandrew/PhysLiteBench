"""Validate fluid-coexistence revision 1; regenerate calibration only with --generate."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['fluid-coexistence']


def module(path):
    spec = importlib.util.spec_from_file_location('candidate_'+str(abs(hash(str(path)))), path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def scale_for(name, experiments, truth):
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
    sigma = .003*max(np.max(np.abs(noiseless)), 1e-10)
    seed = 9212
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
    parser.add_argument('--output', type=Path, default=ROOT/'jobs/fluid-coexistence-validation/summary.json')
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
    from scipy.integrate import quad
    oracle = module(ROOT/'tasks/fluid-coexistence/solution/model.py')
    attraction = ref.TRUE_PARAMETER
    comparisons, areas, tangent_gaps, stability = [], [], [], []
    contacts = {}
    for temperature in [.6, .64, .72, .8, .86]:
        liquid, vapor, p = oracle.coexistence(temperature, attraction)
        independent = ref.phase_contacts(temperature, attraction)
        comparisons.append(float(np.max(abs(np.array([liquid, vapor, p])-independent))))
        area = quad(lambda v: temperature/(v-1)-attraction/v**2-p,
                    liquid, vapor, epsabs=1e-12, epsrel=1e-12)[0]
        areas.append(abs(area))
        grid = np.geomspace(1.001, max(60., 2*vapor), 2000)
        free_energy = ref.free_energy(grid, temperature, attraction)
        tangent = ref.free_energy(liquid, temperature, attraction)-p*(grid-liquid)
        tangent_gaps.append(float(np.min(free_energy-tangent)))
        for v in [liquid, vapor]:
            stability.append(float(temperature/(v-1)**2-2*attraction/v**3))
        contacts[str(temperature)] = {'liquid': liquid, 'vapor': vapor, 'pressure': p}
    assert max(comparisons) < 1e-7 and max(areas) < 1e-10
    assert min(tangent_gaps) >= -1e-10 and min(stability) > 0
    for name, es in ref.hidden_inputs().items():
        liquid, vapor, _ = ref.phase_contacts(es[0]['temperature'], attraction)
        assert min(e['volume'] for e in es) > liquid
        assert max(e['volume'] for e in es) < vapor
    # Stable single-phase branches outside the coexistence interval must retain
    # the original EOS, while every point between contacts has one pressure.
    model = oracle_class()
    model.attraction = attraction
    stable_error, plateau_error = 0., 0.
    for temperature in [.64, .8]:
        liquid, vapor, p = oracle.coexistence(temperature, attraction)
        es = [dict(temperature=temperature, volume=float(v)) for v in [1.1, liquid*.999, vapor*1.001, 50.]]
        truth = np.array([temperature/(e['volume']-1)-attraction/e['volume']**2 for e in es])
        stable_error = max(stable_error, float(np.max(abs(model.predict(es)-truth))))
        es = [dict(temperature=temperature, volume=float(v)) for v in np.linspace(liquid, vapor, 25)]
        plateau_error = max(plateau_error, float(np.max(abs(model.predict(es)-p))))
    assert stable_error < 1e-12 and plateau_error < 1e-12
    critical_t = 8*attraction/27
    liquid, vapor, p = oracle.coexistence(critical_t*.9999, attraction)
    assert abs(liquid-3.) < .08 and abs(vapor-3.) < .08
    assert abs(p-attraction/27) < .0001
    critical_samples = [dict(temperature=critical_t*(1-1e-10), volume=float(v))
                        for v in np.linspace(2.9999, 3.0001, 11)]
    critical_prediction = model.predict(critical_samples)
    assert np.isfinite(critical_prediction).all()
    critical_pressure_error = float(np.max(abs(critical_prediction-attraction/27)))
    assert critical_pressure_error < 1e-9
    return {'near_critical_pressure_error': critical_pressure_error,
            'max_independent_contact_error': max(comparisons),
            'max_numeric_maxwell_area': max(areas),
            'minimum_free_energy_above_tangent': min(tangent_gaps),
            'minimum_endpoint_free_energy_curvature': min(stability),
            'single_phase_pressure_error': stable_error,
            'coexistence_plateau_error': plateau_error,
            'near_critical_volumes': [liquid, vapor],
            'phase_contacts': contacts}



if __name__ == '__main__':
    main()
