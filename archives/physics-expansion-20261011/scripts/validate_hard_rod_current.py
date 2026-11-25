"""Validate hard-rod current r1; --generate intentionally replaces data."""
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
    sigma = .002
    if args.generate:
        rng = np.random.default_rng(611501)
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
    report = {'revision': 1, 'calibration_seed': 611501, 'noise_seed': 611509, 'controls': {}, 'groups': {}}
    for name, model in models.items():
        residual = (model.predict(calibration)-[r['value'] for r in records])/sigma
        report['controls'][name] = dict(speed=model.speed, chi2=float(residual@residual)/(len(records)-1))
        assert report['controls'][name]['chi2'] < 1.5
        assert abs(model.speed/speed-1) < .03
    rng = np.random.default_rng(611509)
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
    rng = np.random.default_rng(611513)
    cases = [ref.experiment(density=float(rng.uniform(.1, .6)), length=float(rng.uniform(0., 1.2)),
                           right_fraction=float(rng.uniform(.25, .75)), counting_field=float(rng.uniform(-4., 4.)))
             for _ in range(128)]
    cases += [ref.experiment(density=r, length=a, right_fraction=p, counting_field=f)
              for r in [.1, .6] for a in [0., 1.2] for p in [.25, .75] for f in [-4., 0., 4.]]
    physical = oracle.predict_at(cases, speed)
    numeric = ref.predict(cases, speed, tolerance=1e-14)
    assert max(abs(physical-numeric)) < 1e-9
    moment_error = 0.
    for e in cases:
        reflected = dict(e, right_fraction=1-e['right_fraction'], counting_field=-e['counting_field'])
        assert abs(oracle.response(e)-oracle.response(reflected)) < 1e-12
        assert abs(oracle.response(dict(e, counting_field=0.))) < 1e-12
        assert abs(ref.response(e, 1.3)-ref.response(e, 1.)*1.3) < 1e-9
        delta = 3e-4
        fp, fm = [oracle.response(dict(e, counting_field=f)) for f in [delta, -delta]]
        mean = oracle.response(dict(e, statistic='mean'))
        var = oracle.response(dict(e, statistic='variance'))
        assert var > 0.
        moment_error = max(moment_error, abs((fp-fm)/(2*delta)-mean), abs((fp+fm)/delta**2-var))
        assert oracle.response(e) >= e['counting_field']*mean-1e-12
    assert moment_error < 2e-7
    microscopic = []
    for e in [ref.experiment(), ref.experiment(density=.6, length=1.2, right_fraction=.3, counting_field=-4.),
              ref.experiment(density=.3, length=.8, right_fraction=.7, counting_field=4.),
              ref.experiment(density=.6, length=1.2, right_fraction=.7, counting_field=-3.)]:
        values = [ref.finite_time_scgf(e, speed, t) for t in [16., 32., 64.]]
        refined = ref.finite_time_scgf(e, speed, 64., tail_width=20, minimum_width=120)
        asymptotic = ref.response(e, speed)
        extrapolated = 2*values[2]-values[1]
        assert abs(refined-values[2]) < 1e-10
        assert abs(extrapolated-asymptotic) < 1e-6
        zero = ref.finite_time_scgf(dict(e, counting_field=0.), speed, 32.)
        assert abs(zero) < 1e-12
        microscopic.append(dict(input=e, times=[16, 32, 64], finite_time_values=values,
                                extrapolated=extrapolated, reference=asymptotic,
                                sum_refinement_error=abs(refined-values[2]), normalization_log_error=abs(zero)))
    report.update(domain_cases=len(cases), domain_reference_max_absolute_difference=float(max(abs(physical-numeric))),
                  moment_derivative_max_absolute_error=moment_error, microscopic_checks=microscopic,
                  invariants=['zero bias', 'reflection', 'speed scaling', 'positive variance', 'mean and variance derivatives'],
                  wall_seconds=time.monotonic()-started)
    (ROOT/'results/expansion-hard-rod-current-r1-validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
