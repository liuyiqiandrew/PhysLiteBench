"""Validate finite-time annihilating-gas survival; --generate replaces data."""
import argparse
import importlib.util
import itertools
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import gammainc, ive

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/annihilating-gas-target'


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
    baseline = load('baseline', ROOT/'scripts/annihilating_gas_target_baseline.py')
    speed, sigma = ref.TRUE_PARAMETER, .001
    calibration = ref.calibration_inputs()*8
    means = ref.predict(calibration, speed)
    assert max(abs(oracle.predict_at(calibration, speed)-means)) < 1e-12
    assert max(abs(baseline.predict_at(calibration, speed)-means)) < 1e-12
    if args.generate:
        rng = np.random.default_rng(612001)
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
    report = dict(revision=1, calibration_seed=612001, noise_seed=612009, controls={}, groups={})
    for name, model in models.items():
        residual = (model.predict(calibration)-[r['value'] for r in records])/sigma
        report['controls'][name] = dict(speed=model.speed, chi2=float(residual@residual)/(len(records)-1))
        assert report['controls'][name]['chi2'] < 1.5 and abs(model.speed/speed-1) < .03
    rng = np.random.default_rng(612009)
    fits, chis = [], []
    for _ in range(256):
        noisy = [dict(r, value=float(y)) for r, y in zip(records, means+rng.normal(0, sigma, len(means)))]
        model = oracle.Model().fit(noisy)
        fits.append(model.speed)
        residual = (model.predict(calibration)-[r['value'] for r in noisy])/sigma
        chis.append(float(residual@residual)/(len(noisy)-1))
    report.update(noise_draws=256, noise_parameter_range=[min(fits), max(fits)], noise_max_chi2=max(chis))
    assert max(chis) < 1.5 and max(abs(np.asarray(fits)/speed-1)) < .03
    for name, rows in ref.hidden_inputs().items():
        expected = np.asarray(truth[name]['values'])
        fresh = ref.predict(rows, speed)
        refined = ref.predict(rows, speed, samples=4096, radius=.99)
        physical = oracle.predict_at(rows, speed)
        assert np.max(abs(fresh-expected)) < 1e-12
        assert error(fresh, refined) < 1e-10 and error(physical, refined) < 1e-10
        errors = {key: error(m.predict(rows), expected) for key, m in models.items()}
        noise_oracle = max(error(oracle.predict_at(rows, p), expected) for p in fits)
        noise_shortcut = min(error(baseline.predict_at(rows, p), expected) for p in fits)
        assert noise_oracle < .04
        if name != 'shared_anchors':
            assert noise_shortcut > .04
        report['groups'][name] = dict(errors=errors, noise_oracle_max=noise_oracle,
                                     noise_shortcut_min=noise_shortcut,
                                     reference_refinement_error=error(fresh, refined),
                                     oracle_reference_error=error(physical, refined))
    # Every calibration curve strictly decreases with speed: if A=ive0+ive1,
    # d(A**2)/dz=-2*A*ive1/z<0. This proves noiseless identifiability.
    derivative_max = -float('inf')
    recovery_error = 0.
    for v in np.linspace(.8, 1.3, 51):
        x = np.asarray([e['density']*e['time'] for e in ref.calibration_inputs()])
        z = v*x
        derivative_max = max(derivative_max, float(max(-2*(ive(0,z)+ive(1,z))*ive(1,z)/v)))
        noiseless = [dict(r, value=float(y)) for r, y in zip(records, oracle.predict_at(calibration, v))]
        recovery_error = max(recovery_error, abs(oracle.Model().fit(noiseless).speed-v))
        assert recovery_error < 1e-7
    assert derivative_max < 0.
    rng = np.random.default_rng(612013)
    cases = [(ref.experiment(float(rng.uniform(.2,1.2)),float(rng.uniform(0.,1.)),float(rng.uniform(0.,60.))),
              float(rng.uniform(.8,1.3))) for _ in range(96)]
    cases += [(ref.experiment(r,p,t),v) for r in [.2,1.2] for p in [0.,.25,1.]
              for t in [0.,60.] for v in [.8,1.3]]
    domain_error, coefficient_error = 0., 0.
    for e,v in cases:
        physical = oracle.response(e,v)
        numeric = ref.predict([e],v)[0]
        domain_error = max(domain_error,abs(physical-numeric))
        assert -1e-12 <= physical <= 1.+1e-12
        assert oracle.response(dict(e,time=0.),v) == 1.
        assert abs(oracle.response(dict(e,stationary_fraction=1.),v)-1.) < 1e-12
        a = oracle.coefficients(e['stationary_fraction'])
        assert min(a) >= 0. and sum(a) <= 1.+1e-12
        coefficient_error = max(coefficient_error,float(max(abs(a-ref.fft_coefficients(e['stationary_fraction'])))))
    truncation = 2*float(gammainc(257,1.2*1.3*60.))
    assert domain_error < 1e-10 and coefficient_error < 1e-10 and truncation < 1e-40
    # Exact finite microscopic enumeration; average all orderings of fixed
    # noncommensurate gaps, then all independent velocity assignments.
    enumeration_error = 0.
    enumeration_configurations = 0
    for n in [1,3,5]:
        counts = np.zeros(n+1)
        gaps = list(itertools.permutations(np.sqrt([2,3,5,7,11][:n])))
        for gap in gaps:
            positions = np.cumsum(gap)
            for velocity in itertools.product([-1,0,1],repeat=n):
                enumeration_configurations += 1
                if ref.first_hit(positions,velocity,positions[-1]) == n:
                    counts[velocity.count(0)] += 1/len(gaps)
        for p in [.2,.5,.8]:
            value = sum(counts[k]*p**k*((1-p)/2)**(n-k) for k in range(n+1))
            enumeration_error = max(enumeration_error,abs(value-oracle.coefficients(p)[n-1]))
    assert enumeration_error < 1e-12
    rng = np.random.default_rng(612017)
    microscopic = []
    for p,z in [(0.,5.),(.1,15.),(.2,10.),(.25,40.),(.3,20.),(.45,25.),(.6,60.),(.8,40.)]:
        count,hits = 50000,0
        for _ in range(count):
            n = rng.poisson(z)
            positions = np.sort(rng.uniform(0.,z,n))
            velocities = rng.choice([-1,0,1],n,p=[(1-p)/2,p,(1-p)/2])
            hits += ref.first_hit(positions,velocities,z) is not None
        expected = 1-np.sqrt(oracle.response(ref.experiment(1.,p,z),1.))
        observed = hits/count
        stderr = np.sqrt(expected*(1-expected)/count)
        assert abs(observed-expected) < 6*stderr
        microscopic.append(dict(p=p,z=z,trials=count,observed_hit=observed,expected_hit=expected,
                                standard_error=float(stderr),standardized_error=float((observed-expected)/stderr)))
    report.update(calibration_derivative_max=derivative_max, noiseless_recovery_cases=51,
                  noiseless_recovery_max_absolute_error=recovery_error,
                  domain_cases=len(cases),domain_reference_max_absolute_difference=domain_error,
                  coefficient_max_absolute_difference=coefficient_error,truncation_bound=truncation,
                  enumeration_configurations=enumeration_configurations,enumeration_max_absolute_error=enumeration_error,
                  microscopic_seed=612017,microscopic_checks=microscopic,
                  invariants=['unit initial survival','p=1 survival','probability bounds','positive first-hit coefficients'],
                  wall_seconds=time.monotonic()-started)
    (ROOT/'results/expansion-annihilating-gas-target-r1-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
