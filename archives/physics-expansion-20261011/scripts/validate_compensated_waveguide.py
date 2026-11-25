"""Validate compensated waveguide r1; --generate intentionally replaces data."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import mathieu_b

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/compensated-waveguide'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def error(actual, expected):
    return float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))


def constant_gap(e):
    q = e['well']/(2*np.pi**2)
    return np.pi**2*(mathieu_b(e['level']+1, q)-mathieu_b(1, q))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    ref = load('ref', TASK/'tests/reference.py')
    oracle = load('oracle', TASK/'solution/model.py')
    baseline = load('baseline', ROOT/'scripts/compensated_waveguide_baseline.py')
    gain = ref.TRUE_PARAMETER
    calibration = ref.calibration_inputs()*20
    means = gain*np.array([constant_gap(e) for e in calibration])
    assert np.max(abs(oracle.predict_at(calibration, gain)-means)) < 1e-9
    assert np.max(abs(baseline.predict_at(calibration, gain)-means)) < 1e-9
    if args.generate:
        rng = np.random.default_rng(611301)
        records = [dict(input=e, value=float(y), sigma=.02)
                   for e, y in zip(calibration, means+rng.normal(0, .02, len(means)))]
        for part in ['environment', 'tests']:
            (TASK/part/'data/calibration.json').write_text(json.dumps(records, indent=2)+'\n')
        truth = {name:dict(inputs=rows, values=ref.predict(rows, gain).tolist())
                 for name, rows in ref.hidden_inputs().items()}
        (TASK/'tests/data/hidden_truth.json').write_text(json.dumps(truth, indent=2)+'\n')
    records = json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records == json.loads((TASK/'tests/data/calibration.json').read_text())
    truth = json.loads((TASK/'tests/data/hidden_truth.json').read_text())
    models = {name:module.Model().fit(records) for name, module in [('oracle', oracle), ('shortcut', baseline)]}
    report = {'revision':1, 'calibration_seed':611301, 'noise_seed':611307, 'controls':{}, 'groups':{}}
    for name, model in models.items():
        residual = (model.predict(calibration)-[r['value'] for r in records])/.02
        report['controls'][name] = dict(gain=model.gain, chi2=float(residual@residual)/(len(records)-1))
        assert report['controls'][name]['chi2'] < 1.5
        assert abs(model.gain/gain-1) < .03
    rng = np.random.default_rng(611307)
    fits, chis = [], []
    basis = means/gain
    for _ in range(256):
        values = means+rng.normal(0, .02, len(means))
        fitted = float(basis@values/(basis@basis))
        fits.append(fitted)
        residual = (basis*fitted-values)/.02
        chis.append(float(residual@residual)/(len(values)-1))
    report.update(noise_draws=256, noise_parameter_range=[min(fits), max(fits)], noise_max_chi2=max(chis))
    assert max(chis) < 1.5
    assert max(abs(np.asarray(fits)/gain-1)) < .03
    for name, rows in ref.hidden_inputs().items():
        expected = np.asarray(truth[name]['values'])
        fresh = ref.predict(rows, gain)
        refined = ref.predict(rows, gain, epsilon=.015, modes=64, transverse=11)
        physical = oracle.predict_at(rows, gain)
        refined_oracle = gain*np.array([oracle.gap(e, modes=120, points=480) for e in rows])
        assert np.max(abs(fresh-expected)) < 1e-6
        assert error(fresh, refined) < .0001
        assert error(physical, refined) < .0001
        assert np.max(abs(physical-refined_oracle)) < 1e-8
        shortcut = baseline.predict_at(rows, gain)
        errors = {key:error((physical if key=='oracle' else shortcut)*m.gain/gain, expected)
                  for key, m in models.items()}
        noise_oracle = max(error(physical*p/gain, expected) for p in fits)
        noise_shortcut = min(error(shortcut*p/gain, expected) for p in fits)
        assert noise_oracle < .04
        if name != 'constant_width':
            assert noise_shortcut > .04
        report['groups'][name] = dict(errors=errors, noise_oracle_max=noise_oracle, noise_shortcut_min=noise_shortcut,
                                     reference_refinement_error=error(fresh, refined), oracle_reference_error=error(physical, refined),
                                     oracle_refinement_max=float(max(abs(physical-refined_oracle))),
                                     min_gap=float(min(expected)))
    rng = np.random.default_rng(611311)
    cases = [ref.experiment(amplitude=float(rng.uniform(0, .8)), harmonic=float(rng.uniform(0, .5)),
                           phase=float(rng.uniform(0, 2*np.pi)), well=float(rng.uniform(-20, 20)),
                           level=int(rng.integers(1, 4))) for _ in range(24)]
    cases.extend([ref.experiment(amplitude=.8, harmonic=.5, phase=ph, well=w, level=n)
                  for ph in [0., np.pi/2, np.pi] for w in [-20., 20.] for n in [1, 3]])
    physical = oracle.predict_at(cases, 1.)
    finite = ref.predict(cases, 1., epsilon=.015, modes=64, transverse=11)
    assert error(physical, finite) < .0001
    reflection_error = 0.
    for e, value in zip(cases, physical):
        reflected = dict(e, phase=2*np.pi-e['phase'])
        reflection_error = max(reflection_error, abs(oracle.gap(reflected)-value))
    assert reflection_error < 1e-8
    # Separate basis and quadrature errors from the width extrapolation.
    e = ref.experiment(amplitude=.8, harmonic=.5, phase=.6, well=-20., level=3)
    widths = [.04, .02, .01]
    convergence = []
    for width in widths:
        a = ref.finite(e, width, modes=48, transverse=7, points=220)
        b = ref.finite(e, width, modes=64, transverse=7, points=320)
        c = ref.finite(e, width, modes=64, transverse=11, points=320)
        d = ref.finite(e, width, modes=64, transverse=15, points=400)
        convergence.append(dict(epsilon=width, gaps=[a, b, c, d]))
        assert abs(c-d)/d < .0001
        assert abs(a-b)/b < .0001
    report.update(domain_cases=len(cases), domain_reference_relative_error=error(physical, finite),
                  domain_reference_max_absolute_difference=float(max(abs(physical-finite))),
                  reflection_error=reflection_error, finite_width_convergence=convergence,
                  wall_seconds=time.monotonic()-started)
    (ROOT/'results/expansion-compensated-waveguide-r1-validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
