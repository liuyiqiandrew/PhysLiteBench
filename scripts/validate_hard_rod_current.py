"""Validate hard-rod current r2; --generate intentionally replaces data."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/hard-rod-current'


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
    baseline = load('baseline', ROOT/'scripts/hard_rod_current_baseline.py')
    speed = ref.TRUE_PARAMETER
    settings = ref.calibration_inputs()
    calibration = settings*5
    means = np.tile(ref.predict(settings, speed), 5)
    assert np.max(abs(oracle.predict_at(calibration, speed)-means)) < 1e-12
    assert np.max(abs(baseline.predict_at(calibration, speed)-means)) < 1e-12
    sigma = .001
    if args.generate:
        rng = np.random.default_rng(611701)
        records = [dict(input=e, value=float(y), sigma=sigma)
                   for e, y in zip(calibration, means+rng.normal(0, sigma, len(means)))]
        for part in ['environment', 'tests']:
            (TASK/part/'data/calibration.json').write_text(json.dumps(records, indent=2)+'\n')
        truth = {name: dict(inputs=rows, values=ref.predict(rows, speed).tolist())
                 for name, rows in ref.hidden_inputs().items()}
        (TASK/'tests/data/hidden_truth.json').write_text(json.dumps(truth, indent=2)+'\n')
    records = json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records == json.loads((TASK/'tests/data/calibration.json').read_text())
    truth = json.loads((TASK/'tests/data/hidden_truth.json').read_text())
    models = {name: module.Model().fit(records) for name, module in [('oracle', oracle), ('shortcut', baseline)]}
    report = {'revision': 2, 'calibration_seed': 611701, 'noise_seed': 611709, 'controls': {}, 'groups': {}}
    for name, model in models.items():
        residual = (model.predict(calibration)-[r['value'] for r in records])/sigma
        report['controls'][name] = dict(speed=model.speed, chi2=float(residual@residual)/(len(records)-1))
        assert report['controls'][name]['chi2'] < 1.5
        assert abs(model.speed/speed-1) < .03
    rng = np.random.default_rng(611709)
    fits, chis = [], []
    basis = means/speed
    for _ in range(256):
        values = means+rng.normal(0, sigma, len(means))
        fitted = float(basis@values/(basis@basis))
        fits.append(fitted)
        residual = (basis*fitted-values)/sigma
        chis.append(float(residual@residual)/(len(values)-1))
    report.update(noise_draws=256, noise_parameter_range=[min(fits), max(fits)], noise_max_chi2=max(chis))
    assert max(chis) < 1.5
    assert max(abs(np.asarray(fits)/speed-1)) < .03
    for name, rows in ref.hidden_inputs().items():
        expected = np.asarray(truth[name]['values'])
        fresh = ref.predict(rows, speed)
        refined = ref.predict(rows, speed, tolerance=1e-14)
        physical = oracle.predict_at(rows, speed)
        assert np.max(abs(fresh-expected)) < 1e-12
        assert error(fresh, refined) < 1e-8
        assert error(physical, refined) < 1e-8
        shortcut = baseline.predict_at(rows, speed)
        errors = {key: error((physical if key == 'oracle' else shortcut)*m.speed/speed, expected)
                  for key, m in models.items()}
        noise_oracle = max(error(physical*p/speed, expected) for p in fits)
        noise_shortcut = min(error(shortcut*p/speed, expected) for p in fits)
        assert noise_oracle < .04
        if name != 'shared_anchors':
            assert noise_shortcut > .04
        report['groups'][name] = dict(errors=errors, noise_oracle_max=noise_oracle,
                                     noise_shortcut_min=noise_shortcut,
                                     reference_refinement_error=error(fresh, refined),
                                     oracle_reference_error=error(physical, refined))
    rng = np.random.default_rng(611713)
    cases = [ref.experiment(density=float(rng.uniform(.1, .6)), length=float(rng.uniform(.4, 1.2)),
                           contrast=float(rng.uniform(0., .75)), counting_field=float(rng.uniform(-4., 4.)))
             for _ in range(96)]
    cases += [ref.experiment(density=r, length=a, contrast=d, counting_field=f)
              for r in [.1, .6] for a in [.4, 1.2] for d in [0., .75] for f in [-4., 0., 4.]]
    physical = oracle.predict_at(cases, speed)
    numeric = ref.predict(cases, speed, tolerance=1e-14)
    assert max(abs(physical-numeric)) < 1e-8
    variance_error = 0.
    for e in cases:
        assert abs(oracle.response(e)-oracle.response(dict(e, counting_field=-e['counting_field']))) < 1e-12
        assert abs(oracle.response(dict(e, counting_field=0.))) < 1e-12
        delta = 1e-4
        variance = oracle.response(dict(e, statistic='variance'))
        estimate = 2*oracle.response(dict(e, counting_field=delta))/delta**2
        variance_error = max(variance_error, abs(estimate-variance))
        assert oracle.response(e) >= baseline.response(e)-1e-12
    # The true correction is cubic in absolute field; finite second differences
    # therefore have O(delta), not O(delta**2), truncation error.
    assert variance_error < 3e-5
    microscopic = []
    for rho, mean, contrast, field in [(.5, 1., .7, 2.), (.6, 1.2, .75, 2.),
                                       (.6, 1.2, .75, 4.), (.3, .8, .5, 4.)]:
        values = [ref.finite_positive_mgf(rho, mean, contrast, field, t, speed) for t in [16., 32., 64.]]
        extended = ref.finite_positive_mgf(rho, mean, contrast, field, 128., speed)
        expected = ref.contraction(rho, mean, contrast, field, speed)
        assert abs(2*values[2]-values[1]-expected) < 1e-6
        assert abs(2*extended-values[2]-expected) < 1e-6
        microscopic.append(dict(density=rho, length=mean, contrast=contrast, counting_field=field,
                                times=[16, 32, 64, 128], values=values+[extended], reference=expected,
                                extrapolation_error=abs(2*extended-values[2]-expected)))
    report.update(domain_cases=len(cases), domain_reference_max_absolute_difference=float(max(abs(physical-numeric))),
                  finite_difference_variance_max_error=variance_error, microscopic_checks=microscopic,
                  invariants=['zero mean and bias', 'field reflection', 'variance', 'mixture exceeds mean-length SCGF'],
                  wall_seconds=time.monotonic()-started)
    (ROOT/'results/expansion-hard-rod-current-r2-validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
