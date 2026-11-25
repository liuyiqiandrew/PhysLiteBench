"""Validate accelerated detector pair r1; --generate intentionally replaces data."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/accelerated-detector-pair'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def error(actual, expected):
    return float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    ref = load('ref', TASK/'tests/reference.py')
    oracle = load('oracle', TASK/'solution/model.py')
    baseline = load('baseline', ROOT/'scripts/accelerated_detector_pair_baseline.py')
    coupling = ref.TRUE_PARAMETER
    settings = ref.calibration_inputs()
    calibration = settings*8
    means = np.tile(ref.predict(settings, coupling), 8)
    assert np.max(abs(oracle.predict_at(calibration, coupling)-means)) < 1e-12
    assert np.max(abs(baseline.predict_at(calibration, coupling)-means)) < 1e-12
    sigma = 2e-8
    if args.generate:
        rng = np.random.default_rng(611401)
        records = [dict(input=e, value=float(y), sigma=sigma)
                   for e, y in zip(calibration, means+rng.normal(0, sigma, len(means)))]
        for part in ['environment', 'tests']:
            (TASK/part/'data/calibration.json').write_text(json.dumps(records, indent=2)+'\n')
        truth = {name:dict(inputs=rows, values=ref.predict(rows, coupling).tolist())
                 for name, rows in ref.hidden_inputs().items()}
        (TASK/'tests/data/hidden_truth.json').write_text(json.dumps(truth, indent=2)+'\n')
    records = json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records == json.loads((TASK/'tests/data/calibration.json').read_text())
    truth = json.loads((TASK/'tests/data/hidden_truth.json').read_text())
    models = {name:module.Model().fit(records) for name, module in [('oracle', oracle), ('shortcut', baseline)]}
    report = {'revision':1, 'calibration_seed':611401, 'noise_seed':611409, 'controls':{}, 'groups':{}}
    for name, model in models.items():
        residual = (model.predict(calibration)-[r['value'] for r in records])/sigma
        report['controls'][name] = dict(coupling_squared=model.coupling_squared, chi2=float(residual@residual)/(len(records)-1))
        assert report['controls'][name]['chi2'] < 1.5
        assert abs(model.coupling_squared/coupling-1) < .03
    rng = np.random.default_rng(611409)
    fits, chis = [], []
    basis = means/coupling
    for _ in range(256):
        values = means+rng.normal(0, sigma, len(means))
        fitted = float(basis@values/(basis@basis))
        fits.append(fitted)
        residual = (basis*fitted-values)/sigma
        chis.append(float(residual@residual)/(len(values)-1))
    report.update(noise_draws=256, noise_parameter_range=[min(fits), max(fits)], noise_max_chi2=max(chis))
    assert max(chis) < 1.5
    assert max(abs(np.asarray(fits)/coupling-1)) < .03
    for name, rows in ref.hidden_inputs().items():
        expected = np.asarray(truth[name]['values'])
        fresh = ref.predict(rows, coupling)
        refined = ref.predict(rows, coupling, tolerance=1e-13, shift=.63)
        physical = oracle.predict_at(rows, coupling)
        assert np.max(abs(fresh-expected)) < 1e-12
        assert error(fresh, refined) < 1e-7
        assert error(physical, refined) < 1e-7
        shortcut = baseline.predict_at(rows, coupling)
        errors = {key:error((physical if key=='oracle' else shortcut)*m.coupling_squared/coupling, expected)
                  for key, m in models.items()}
        noise_oracle = max(error(physical*p/coupling, expected) for p in fits)
        noise_shortcut = min(error(shortcut*p/coupling, expected) for p in fits)
        assert noise_oracle < .04
        if name != 'shared_anchors':
            assert noise_shortcut > .04
        report['groups'][name] = dict(errors=errors, noise_oracle_max=noise_oracle, noise_shortcut_min=noise_shortcut,
                                     reference_refinement_error=error(fresh, refined), oracle_reference_error=error(physical, refined),
                                     min_rate=float(min(expected)))
    rng = np.random.default_rng(611413)
    cases = [ref.experiment(preparation=str(rng.choice(['single', 'symmetric', 'antisymmetric'])),
                           gap=float(rng.uniform(.5, 2.)), acceleration=float(rng.uniform(0., 10.)),
                           separation=float(rng.uniform(.1, 3.))) for _ in range(64)]
    cases.extend([ref.experiment(preparation=p, gap=w, acceleration=a, separation=d)
                  for p in ['symmetric', 'antisymmetric'] for w in [.5, 2.] for a in [0., .01, 10.] for d in [.1, 3.]])
    physical = oracle.predict_at(cases, 1.)
    numeric = ref.predict(cases, 1., tolerance=1e-12, shift=.8)
    assert np.max(abs(physical-numeric)) < 1e-9
    assert min(physical) >= 0.
    identity_error = 0.
    for e in cases:
        s = oracle.response(dict(e, preparation='symmetric'))
        a = oracle.response(dict(e, preparation='antisymmetric'))
        local = oracle.response(dict(e, preparation='single'))
        identity_error = max(identity_error, abs(s+a-2*local))
        scaled = dict(e, gap=e['gap']*1.7, acceleration=e['acceleration']*1.7, separation=e['separation']/1.7)
        assert abs(oracle.response(scaled)-1.7*oracle.response(e)) < 1e-12
    assert identity_error < 1e-12
    # Directly integrate both spectral signs at moderate acceleration.
    kms_error = 0.
    for a in [1., 4., 10.]:
        for gap in [.5, 1.5]:
            for distance in [0., .7, 2.]:
                down = ref.spectrum(gap, a, distance, tolerance=1e-13)
                up = ref.spectrum(-gap, a, distance, tolerance=1e-13)
                kms_error = max(kms_error, abs(up-down*np.exp(-2*np.pi*gap/a)))
    assert kms_error < 1e-10
    report.update(domain_cases=len(cases), domain_reference_max_absolute_difference=float(max(abs(physical-numeric))),
                  pair_sum_error=identity_error, direct_spectral_kms_error=kms_error,
                  invariants=['nonnegative resolved rates', 'pair sum equals twice single rate', 'dimensional scaling', 'direct Fourier KMS'],
                  wall_seconds=time.monotonic()-started)
    (ROOT/'results/expansion-accelerated-detector-pair-r1-validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
