"""Validate triad-photon-phase revision 1; regenerate calibration only with --generate."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['triad-photon-phase']


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
    sigma = .006*max(np.max(np.abs(noiseless)), 1e-10)
    seed = 9327
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
    parser.add_argument('--output', type=Path, default=ROOT/'jobs/triad-photon-phase-validation/summary.json')
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
    from itertools import product
    from math import factorial
    model = oracle_class(); model.rotation_gain = ref.TRUE_PARAMETER
    baseline = baseline_class(); baseline.rotation_gain = model.rotation_gain
    probability = model.predict.__func__.__globals__['probability']
    all_es = [e for es in ref.hidden_inputs().values() for e in es]
    reference_error = float(np.max(abs(model.predict(all_es)-ref.predict(all_es,model.rotation_gain))))
    assert reference_error < 1e-12
    rng = np.random.default_rng(88312)
    gauge_error = 0.
    normalization_error = 0.
    minimum_probability = 1.
    minimum_gram_eigenvalue = 1.
    pairwise_error = 0.
    alternative_state_error = 0.
    for n in [2,3]:
        ports = np.arange(n)
        unitary = np.exp(2j*np.pi*np.outer(ports,ports)/n)/np.sqrt(n)
        counts = [list(c) for c in product(range(n+1),repeat=n) if sum(c)==n]
        for k in range(30):
            gain = rng.uniform(.6,1.4)
            lengths = rng.uniform(0.,3.5,n)
            jones = np.column_stack([np.cos(gain*lengths),np.sin(gain*lengths)])
            gram = jones.conj()@jones.T
            rephased = jones*np.exp(1j*rng.uniform(-np.pi,np.pi,n))[:,None]
            rephased_gram = rephased.conj()@rephased.T
            values = np.array([probability(unitary,gram,c) for c in counts])
            other = np.array([probability(unitary,rephased_gram,c) for c in counts])
            gauge_error = max(gauge_error,float(np.max(abs(values-other))))
            magnitude = np.abs(gram)
            shortcut = np.array([probability(unitary,magnitude,c) for c in counts])
            eigenvalues,eigenvectors = np.linalg.eigh(magnitude)
            minimum_gram_eigenvalue = min(minimum_gram_eigenvalue,float(min(eigenvalues)))
            alternative = eigenvectors@np.diag(np.sqrt(np.maximum(eigenvalues,0.)))
            alternative_probabilities = ref.fock_distribution(unitary,alternative)
            alternative_state_error = max(alternative_state_error,max(abs(shortcut[i]-alternative_probabilities[tuple(c)]) for i,c in enumerate(counts)))
            normalization_error = max(normalization_error,abs(sum(values)-1),abs(sum(shortcut)-1))
            minimum_probability = min(minimum_probability,float(min(values)),float(min(shortcut)))
            if n==2:
                pairwise_error = max(pairwise_error,float(np.max(abs(values-shortcut))))
    assert gauge_error < 1e-12 and normalization_error < 1e-12 and minimum_probability > -1e-12
    assert minimum_gram_eigenvalue > -1e-12 and alternative_state_error < 1e-12 and pairwise_error < 1e-12
    unitary = np.exp(2j*np.pi*np.outer(np.arange(3),np.arange(3))/3)/np.sqrt(3)
    counts = [list(c) for c in product(range(4),repeat=3) if sum(c)==3]
    # Orthogonal internal states give a classical multinomial distribution.
    classical_error = max(abs(probability(unitary,np.eye(3),c)-factorial(3)/np.prod([factorial(v) for v in c])/27) for c in counts)
    identical_one_each_error = abs(probability(unitary,np.ones((3,3)),[1,1,1])-1/3)
    identical_suppression_error = abs(probability(unitary,np.ones((3,3)),[2,1,0]))
    assert classical_error < 1e-12 and identical_one_each_error < 1e-12 and identical_suppression_error < 1e-12
    # Three real polarization directions can have a negative invariant cycle.
    angles = np.array([0.,np.pi/3,2*np.pi/3])
    jones = np.column_stack([np.cos(angles),np.sin(angles)])
    gram = jones@jones.T
    cycle = float(gram[0,1]*gram[1,2]*gram[2,0])
    physical = probability(unitary,gram,[1,1,1])
    shortcut = probability(unitary,np.abs(gram),[1,1,1])
    triad_example_error = abs((shortcut-physical)-1/9)
    assert abs(cycle+1/8) < 1e-12 and triad_example_error < 1e-12
    return dict(fock_reference_error=reference_error, jones_rephasing_error=gauge_error,
                normalization_error=float(normalization_error), minimum_probability=minimum_probability,
                minimum_magnitude_gram_eigenvalue=minimum_gram_eigenvalue,
                alternative_internal_state_reference_error=float(alternative_state_error),
                pairwise_closure_agreement=pairwise_error, distinguishable_multinomial_error=float(classical_error),
                identical_one_each_error=float(identical_one_each_error), identical_suppression_error=float(identical_suppression_error),
                triad_cycle=cycle, triad_example_physical_probability=physical,
                triad_example_shortcut_probability=shortcut, triad_example_error=float(triad_example_error))


if __name__ == '__main__':
    main()
