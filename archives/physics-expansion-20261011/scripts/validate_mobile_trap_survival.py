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
        rng = np.random.default_rng(611801)
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
    report = dict(revision=1, calibration_seed=611801, noise_seed=611809, controls={}, groups={})
    for name, model in models.items():
        residual = (model.predict(calibration)-[r['value'] for r in records])/sigma
        report['controls'][name] = dict(trap_diffusivity=model.trap_diffusivity,
                                       chi2=float(residual@residual)/(len(records)-1))
        assert report['controls'][name]['chi2'] < 1.5
        assert abs(model.trap_diffusivity/parameter-1) < .03
    rng = np.random.default_rng(611809)
    fits, chis = [], []
    basis = means/np.sqrt(parameter)
    for _ in range(256):
        values = means+rng.normal(0, sigma, len(means))
        amplitude = float(basis@values/(basis@basis))
        fits.append(amplitude**2)
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
        if name != 'stationary_anchors':
            assert noise_shortcut > .04
        report['groups'][name] = dict(errors=errors, noise_oracle_max=noise_oracle,
                                     noise_shortcut_min=noise_shortcut,
                                     reference_refinement_error=error(fresh, refined),
                                     oracle_reference_error=error(physical, refined))
    rng = np.random.default_rng(611813)
    cases = [(ref.experiment(float(rng.uniform(.1, 1.)), float(rng.uniform(0., 4.))),
              float(rng.uniform(.5, 1.5))) for _ in range(96)]
    cases += [(ref.experiment(r, d), b) for r in [.1, 1.] for d in [0., 4.] for b in [.5, 1.5]]
    domain_error, gap_error, max_final_gap = 0., 0., 0.
    for e, diffusivity in cases:
        expected = oracle.response(e, diffusivity)
        domain_error = max(domain_error, abs(ref.predict([e], diffusivity)[0]-expected))
        previous_gap = None
        for t in [1., 1e6, 1e12, 1e24, 1e36]:
            lo, hi = ref.bounds(e, diffusivity, t)
            analytic_gap = 3*(np.pi**2*e['density']**2*e['target_diffusivity']/4)**(1/3)*t**(-1/6)
            assert abs(lo-expected) < 1e-12 and hi >= lo
            gap_error = max(gap_error, abs(hi-lo-analytic_gap))
            if previous_gap is not None:
                assert hi-lo <= previous_gap+1e-12
            previous_gap = hi-lo
        max_final_gap = max(max_final_gap, (hi-lo)/expected)
        # Independent Brownian scaling of the readout and density linearity.
        assert abs(oracle.response(dict(e, density=e['density']/2), diffusivity)*2-expected) < 1e-12
        assert abs(oracle.response(e, diffusivity/4)*2-expected) < 1e-12
    assert domain_error < 1e-12 and gap_error < 1e-9 and max_final_gap < 1e-5
    # The confinement bound follows from a positive ground-state martingale.
    # Check it against the independently convergent Dirichlet spectral series.
    q = np.geomspace(.2, 20., 80)
    k = np.arange(80)
    confined = 4/np.pi*np.sum((-1.)**k[:, None]/(2*k[:, None]+1)*
                             np.exp(-(2*k[:, None]+1)**2*q[None, :]), axis=0)
    assert np.all(confined >= np.exp(-q)-1e-14) and np.all(confined <= 1.+1e-14)
    report.update(domain_cases=len(cases), domain_reference_max_absolute_difference=domain_error,
                  bounds_optimization_max_absolute_error=gap_error, bounds_times=[1., 1e6, 1e12, 1e24, 1e36],
                  bounds_final_max_relative_gap=max_final_gap, confinement_series_cases=len(q),
                  invariants=['density linearity', 'Brownian time scaling', 'stationary target equality', 'survival squeeze'],
                  wall_seconds=time.monotonic()-started)
    (ROOT/'results/expansion-mobile-trap-survival-r1-validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
