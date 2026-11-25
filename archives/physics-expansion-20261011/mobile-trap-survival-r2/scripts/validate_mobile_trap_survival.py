"""Validate mobile-trap survival; --generate intentionally replaces data."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/mobile-trap-survival'


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
    baseline = load('baseline', ROOT/'scripts/mobile_trap_survival_baseline.py')
    parameter, sigma = ref.TRUE_PARAMETER, .001
    calibration = ref.calibration_inputs()*8
    means = ref.predict(calibration, parameter)
    assert max(abs(oracle.predict_at(calibration, parameter)-means)) < 1e-12
    assert max(abs(baseline.predict_at(calibration, parameter)-means)) < 1e-12
    if args.generate:
        rng = np.random.default_rng(611901)
        records = [dict(input=e, value=float(y), sigma=sigma)
                   for e, y in zip(calibration, means+rng.normal(0, sigma, len(means)))]
        for part in ['environment', 'tests']:
            (TASK/part/'data/calibration.json').write_text(json.dumps(records, indent=2)+'\n')
        truth = {name: dict(inputs=rows, values=ref.predict(rows, parameter).tolist())
                 for name, rows in ref.hidden_inputs().items()}
        (TASK/'tests/data/hidden_truth.json').write_text(json.dumps(truth, indent=2)+'\n')
    records = json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records == json.loads((TASK/'tests/data/calibration.json').read_text())
    truth = json.loads((TASK/'tests/data/hidden_truth.json').read_text())
    models = {name: module.Model().fit(records) for name, module in [('oracle', oracle), ('shortcut', baseline)]}
    report = dict(revision=2, calibration_seed=611901, noise_seed=611909, controls={}, groups={})
    for name, model in models.items():
        residual = (model.predict(calibration)-[r['value'] for r in records])/sigma
        report['controls'][name] = dict(mobility=model.mobility,
                                       chi2=float(residual@residual)/(len(records)-1))
        assert report['controls'][name]['chi2'] < 1.5
        assert abs(model.mobility/parameter-1) < .03
    rng = np.random.default_rng(611909)
    fits, chis = [], []
    basis = means/parameter
    for _ in range(256):
        values = means+rng.normal(0, sigma, len(means))
        amplitude = float(basis@values/(basis@basis))
        fits.append(amplitude)
        residual = (basis*amplitude-values)/sigma
        chis.append(float(residual@residual)/(len(values)-1))
    report.update(noise_draws=256, noise_parameter_range=[min(fits), max(fits)], noise_max_chi2=max(chis))
    assert max(chis) < 1.5
    assert max(abs(np.asarray(fits)/parameter-1)) < .03
    for name, rows in ref.hidden_inputs().items():
        expected = np.asarray(truth[name]['values'])
        fresh = ref.predict(rows, parameter)
        refined = ref.predict(rows, parameter, tolerance=1e-13)
        physical = oracle.predict_at(rows, parameter)
        assert np.max(abs(fresh-expected)) < 1e-12
        assert error(fresh, refined) < 1e-10
        assert error(physical, refined) < 1e-10
        errors = {key: error(m.predict(rows), expected) for key, m in models.items()}
        noise_oracle = max(error(oracle.predict_at(rows, p), expected) for p in fits)
        noise_shortcut = min(error(baseline.predict_at(rows, p), expected) for p in fits)
        assert noise_oracle < .04
        if name != 'cold_anchors':
            assert noise_shortcut > .04
        report['groups'][name] = dict(errors=errors, noise_oracle_max=noise_oracle,
                                     noise_shortcut_min=noise_shortcut,
                                     reference_refinement_error=error(fresh, refined),
                                     oracle_reference_error=error(physical, refined))
    rng = np.random.default_rng(611913)
    cases = [(ref.experiment(float(rng.uniform(.1, 1.)), float(rng.uniform(-4., 4.)),
                            float(rng.uniform(0., 2.)), float(rng.uniform(.5, 1.5))),
              float(rng.uniform(.8, 1.3))) for _ in range(96)]
    cases += [(ref.experiment(r, f, t, b), m) for r in [.1, 1.] for f in [-4., 0., 4.]
              for t in [0., 2.] for b in [.5, 1.5] for m in [.8, 1.3]]
    cases += [(ref.experiment(r, 2*r*t, t, 1.), 1.) for r in [.1, .5, 1.] for t in [.1, 1., 2.]]
    domain_error, max_lower_error, max_upper_error = 0., 0., 0.
    for e, mobility in cases:
        expected = oracle.predict_at([e], mobility)[0]
        domain_error = max(domain_error, abs(ref.predict([e], mobility)[0]-expected))
        assert abs(oracle.response(e)-oracle.response(dict(e, force=-e['force']))) < 1e-12
        assert oracle.response(dict(e, force=0.)) == 0.
        assert oracle.response(e) <= baseline.response(e)+1e-12
        previous_width = float('inf')
        for time_value in [1., 1e3, 1e6, 1e9]:
            lower, upper = ref.bounds(e, mobility, time_value)
            assert lower >= -1e-12 and lower <= upper+1e-10
            # Both are bounds on the finite-time normalized log survival,
            # not a finite-time bracket around its limiting coefficient.
            assert upper-lower <= previous_width+1e-10
            previous_width = upper-lower
        max_lower_error = max(max_lower_error, abs(lower-expected))
        max_upper_error = max(max_upper_error, abs(upper-expected))
    assert domain_error < 1e-10 and max_lower_error < .0001 and max_upper_error < .001
    # An independent hitting-probability integral checks the range formula.
    from scipy.integrate import quad
    from scipy.special import ndtr, log_ndtr
    range_error = 0.
    for drift in [0., .001, .2, 2.]:
        for diffusivity in [.5, 1.5]:
            for time_value in [.2, 3., 20.]:
                def tail(x, v):
                    scale = np.sqrt(2*diffusivity*time_value)
                    return ndtr((v*time_value-x)/scale)+np.exp(v*x/diffusivity+log_ndtr((-v*time_value-x)/scale))
                integral = quad(lambda x: tail(x, drift)+tail(x, -drift), 0., np.inf,
                                epsabs=1e-10, epsrel=1e-10)[0]
                range_error = max(range_error, abs(integral-ref.expected_range(drift, diffusivity, time_value)))
    assert range_error < 1e-8
    report.update(domain_cases=len(cases), domain_reference_max_absolute_difference=domain_error,
                  bounds_times=[1., 1e3, 1e6, 1e9], finite_endpoint_max_error=max_lower_error,
                  finite_tube_max_error=max_upper_error, range_integral_cases=24,
                  range_integral_max_absolute_difference=range_error,
                  invariants=['force reflection', 'zero force rate', 'thermal survival enhancement', 'survival bounds converge'],
                  wall_seconds=time.monotonic()-started)
    (ROOT/'results/expansion-mobile-trap-survival-r2-validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
