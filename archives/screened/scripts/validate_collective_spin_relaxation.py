"""Validate collective-spin-relaxation revision 1; regenerate calibration only with --generate."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['collective-spin-relaxation']


def module(path):
    spec = importlib.util.spec_from_file_location('candidate_'+str(abs(hash(str(path)))), path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def scale_for(name, experiments, truth):
    return np.ones_like(truth)


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
    seed = 9413
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
    parser.add_argument('--output', type=Path, default=ROOT/'jobs/collective-spin-relaxation-validation/summary.json')
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
    from scipy.linalg import expm
    oracle_module = module(ROOT/'tasks/collective-spin-relaxation/solution/model.py')
    m = oracle_class(); m.temperature = ref.TRUE_PARAMETER
    experiments = [e for es in ref.hidden_inputs().values() for e in es]+ref.calibration_inputs()
    reference_error = float(np.max(abs(m.predict(experiments)-ref.predict(experiments,m.temperature))))
    assert reference_error < 1e-12
    spin2 = sum((a+b)@(a+b) for a,b in zip(oracle_module.FIRST,oracle_module.SECOND))
    commutators = []
    for op in [.9*oracle_module.EXCHANGE-.8*oracle_module.MAGNETIZATION,
               oracle_module.FIRST[0]+oracle_module.SECOND[0],
               oracle_module.FIRST[1]+oracle_module.SECOND[1]]:
        commutators.append(float(np.max(abs(spin2@op-op@spin2))))
    assert max(commutators) < 1e-12
    # Independently propagate a detailed-balance population generator from up-up.
    generator_error = 0.
    normalization_error = 0.
    dark_population = 0.
    for h in [-1.4,-.3,.3,1.4]:
        energies = np.array([.225-h,.225,.225+h,-.675])
        generator = np.zeros((4,4))
        for a,b in [(0,1),(1,2)]:
            generator[b,a] = np.exp(-(energies[b]-energies[a])/(2*m.temperature))
            generator[a,b] = np.exp(-(energies[a]-energies[b])/(2*m.temperature))
        generator -= np.diag(generator.sum(axis=0))
        p = expm(80*generator)@np.array([1.,0.,0.,0.])
        expected = np.exp(-(energies[:3]-min(energies[:3]))/m.temperature)
        expected /= expected.sum()
        generator_error = max(generator_error,float(np.max(abs(p[:3]-expected))))
        normalization_error = max(normalization_error,float(abs(p.sum()-1)))
        dark_population = max(dark_population,float(abs(p[3])))
    assert generator_error < 1e-12 and normalization_error < 1e-12 and dark_population < 1e-12
    symmetry_es = [ref.experiment(h) for h in [.1,.3,.7,1.5]]
    reverse = [dict(e,field=-e['field']) for e in symmetry_es]
    reflection_error = float(np.max(abs(m.predict(symmetry_es)+m.predict(reverse))))
    assert reflection_error < 1e-12
    for temperature, expected in [(1e-4,1.),(1e6,0.)]:
        m.temperature = temperature
        value = m.predict([ref.experiment(1.5)])[0]
        assert abs(value-expected) < 2e-6
    return dict(analytic_reference_error=reference_error,total_spin_commutator_max=max(commutators),
                rate_generator_stationarity_error=generator_error,rate_generator_normalization_error=normalization_error,
                inaccessible_singlet_population=dark_population,field_reflection_error=reflection_error,
                low_and_high_temperature_limits=True)


if __name__ == '__main__':
    main()
