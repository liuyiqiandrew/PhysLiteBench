"""Validate swirling heat tails r1; --generate intentionally replaces data."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/swirling-heat-tails'


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
    baseline = load('baseline', ROOT/'scripts/swirling_heat_tails_baseline.py')
    mobility = ref.TRUE_PARAMETER
    settings = ref.calibration_inputs()
    calibration = settings*8
    means = np.tile(ref.predict(settings, mobility), 8)
    assert np.max(abs(oracle.predict_at(calibration, mobility)-means)) < 2e-7
    assert np.max(abs(baseline.predict_at(calibration, mobility)-means)) < 2e-7
    sigma = .0004
    if args.generate:
        rng = np.random.default_rng(611601)
        records = [dict(input=e, value=float(y), sigma=sigma)
                   for e, y in zip(calibration, means+rng.normal(0, sigma, len(means)))]
        for part in ['environment', 'tests']:
            (TASK/part/'data/calibration.json').write_text(json.dumps(records, indent=2)+'\n')
        truth = {name: dict(inputs=rows, values=ref.predict(rows, mobility).tolist())
                 for name, rows in ref.hidden_inputs().items()}
        (TASK/'tests/data/hidden_truth.json').write_text(json.dumps(truth, indent=2)+'\n')
    records = json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records == json.loads((TASK/'tests/data/calibration.json').read_text())
    truth = json.loads((TASK/'tests/data/hidden_truth.json').read_text())
    models = {name: module.Model().fit(records) for name, module in [('oracle', oracle), ('shortcut', baseline)]}
    report = {'revision': 1, 'calibration_seed': 611601, 'noise_seed': 611609, 'controls': {}, 'groups': {}}
    for name, model in models.items():
        residual = (model.predict(calibration)-[r['value'] for r in records])/sigma
        report['controls'][name] = dict(mobility=model.mobility, chi2=float(residual@residual)/(len(records)-1))
        assert report['controls'][name]['chi2'] < 1.5
        assert abs(model.mobility/mobility-1) < .03
    rng = np.random.default_rng(611609)
    fits, chis = [], []
    basis = means/mobility
    for _ in range(256):
        values = means+rng.normal(0, sigma, len(means))
        fitted = float(basis@values/(basis@basis))
        fits.append(fitted)
        residual = (basis*fitted-values)/sigma
        chis.append(float(residual@residual)/(len(values)-1))
    report.update(noise_draws=256, noise_parameter_range=[min(fits), max(fits)], noise_max_chi2=max(chis))
    assert max(chis) < 1.5
    assert max(abs(np.asarray(fits)/mobility-1)) < .03
    for name, rows in ref.hidden_inputs().items():
        expected = np.asarray(truth[name]['values'])
        fresh = ref.predict(rows, mobility)
        refined = ref.predict(rows, mobility, step=.003)
        physical = oracle.predict_at(rows, mobility)
        assert np.max(abs(fresh-expected)) < 2e-7
        assert error(fresh, refined) < 1e-5
        assert error(physical, refined) < 1e-5
        shortcut = baseline.predict_at(rows, mobility)
        errors = {key: error((physical if key == 'oracle' else shortcut)*m.mobility/mobility, expected)
                  for key, m in models.items()}
        noise_oracle = max(error(physical*p/mobility, expected) for p in fits)
        noise_shortcut = min(error(shortcut*p/mobility, expected) for p in fits)
        assert noise_oracle < .04
        if name != 'central_anchors':
            assert noise_shortcut > .04
        report['groups'][name] = dict(errors=errors, noise_oracle_max=noise_oracle,
                                     noise_shortcut_min=noise_shortcut,
                                     reference_refinement_error=error(fresh, refined),
                                     oracle_reference_error=error(physical, refined))
    rng = np.random.default_rng(611613)
    cases = [ref.experiment(stiffness=float(rng.uniform(.5, 2.)), drive_ratio=float(rng.uniform(.1, 1.2)),
                           relative_heat=float(rng.uniform(-6., 9.))) for _ in range(96)]
    cases += [ref.experiment(stiffness=k, drive_ratio=r, relative_heat=z)
              for k in [.5, 2.] for r in [.1, 1/np.sqrt(3)-1e-5, 1/np.sqrt(3), 1/np.sqrt(3)+1e-5, 1.2]
              for z in [-6., -1., 1., 3., 9.]]
    physical = oracle.predict_at(cases, mobility)
    numeric = ref.predict(cases, mobility, step=.003)
    assert max(abs(physical-numeric)) < 2e-6
    assert min(physical) > -1e-12
    for e in cases:
        assert abs(oracle.response(dict(e, relative_heat=1.))) < 1e-12
        assert abs(oracle.response(dict(e, stiffness=2*e['stiffness']))-2*oracle.response(e)) < 1e-12
        if e['relative_heat'] < -1.:
            expected = -2*e['stiffness']*e['drive_ratio']**2*e['relative_heat']
            assert abs(oracle.response(e)-expected) < 1e-12
    microscopic = []
    for r in [.15, .3, .8, 1.2]:
        low, high = ref.domain(r, .003)
        for field in [.8*low, .8*high]:
            finite = [ref.finite_time_scgf(r, field, .003, t) for t in [24., 48., 96.]]
            bulk = ref.discrete_scgf(r, field, .003)
            extrapolated = 2*finite[2]-finite[1]
            assert abs(extrapolated-bulk) < 1e-7
            microscopic.append(dict(drive_ratio=r, field=field, times=[24, 48, 96],
                                    finite_time_values=finite, extrapolated=extrapolated, bulk=bulk))
        for field in [low-.02, high+.02]:
            assert not np.isfinite(ref.finite_time_scgf(r, field, .003, 96.))
    report.update(domain_cases=len(cases), domain_reference_max_absolute_difference=float(max(abs(physical-numeric))),
                  microscopic_checks=microscopic,
                  invariants=['nonnegative rate', 'zero rate at mean', 'mobility and stiffness scaling',
                              'reverse linear tail', 'finite path determinant and divergence'],
                  wall_seconds=time.monotonic()-started)
    (ROOT/'results/expansion-swirling-heat-tails-r1-validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
