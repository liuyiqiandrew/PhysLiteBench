"""Validate the finite-mass mechanical-area revision of magnetic-tracer."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/magnetic-tracer'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-data', action='store_true')
    parser.add_argument('--noise-trials', type=int, default=256)
    args = parser.parse_args()
    started = time.monotonic()
    ref = load('reference', TASK / 'tests/reference.py')
    good = load('oracle', TASK / 'solution/model.py')
    bad = load('shortcut', ROOT / 'scripts/magnetic_tracer_baseline.py')
    metadata = json.loads((TASK / 'tests/metadata.json').read_text())
    inputs = ref.calibration_inputs()
    true = ref.TRUE_PARAMETER
    clean = ref.predict(inputs, true)
    sigma = metadata['measurement_sigma']
    def records(values):
        return [dict(input=e, value=float(v), sigma=sigma) for e, v in zip(inputs, values)]
    if args.generate_data:
        measured = records(clean + np.random.default_rng(metadata['calibration_seed']).normal(0, sigma, len(inputs)))
        for part in ['environment', 'tests']:
            (TASK / part / 'data/calibration.json').write_text(json.dumps(measured, indent=2) + '\n')
    measured = json.loads((TASK / 'environment/data/calibration.json').read_text())
    assert measured == json.loads((TASK / 'tests/data/calibration.json').read_text())
    hidden = ref.hidden_inputs()
    truths = {name: ref.predict(es, true) for name, es in hidden.items()}
    def scores(model, data):
        residual = (model.predict(inputs) - np.array([r['value'] for r in data])) / sigma
        errors = {name: float(np.linalg.norm(model.predict(es) - truths[name]) / np.linalg.norm(truths[name]))
                  for name, es in hidden.items()}
        return dict(friction=model.friction, parameter_relative_error=abs(model.friction / true - 1),
                    calibration_chi2=float(residual @ residual) / (len(inputs) - 1), hidden=errors)
    controls = {}
    for name, source in [('oracle', good), ('shortcut', bad)]:
        controls[name] = scores(source.Model().fit(measured), measured)
        assert controls[name]['parameter_relative_error'] < .03 and controls[name]['calibration_chi2'] < 1.5
        errors = list(controls[name]['hidden'].values())
        assert max(errors) < .025 if name == 'oracle' else min(errors) > .025
    print('Fixed-data controls passed', flush=True)
    rng = np.random.default_rng(723801)
    errors, refined, min_covariance, reversal = [], [], [], []
    for gamma in [.5, .8, 1.5]:
        for i in range(12):
            angle = rng.uniform(-np.pi, np.pi)
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            c = rotation @ np.diag(rng.uniform(.2, 2., 2)) @ rotation.T
            e = ref.experiment(float(rng.uniform(.1, 4)), float(rng.uniform(-2, 2)),
                               float(rng.uniform(.5, 1.2)), rng.uniform(.7, 1.8, 2), c)
            if i == 0:
                e = ref.experiment(.1, 2., 1.2, (.7, 1.8), ((1.5, .3), (.3, .5)))
            es = [{**e, 'readout': kind} for kind in ['xx', 'xy', 'yy', 'area']]
            oracle = good.predict_at(es, gamma)
            physical = ref.predict(es, gamma)
            fine = ref.predict(es, gamma, .001)
            errors.append(float(np.max(abs(oracle - physical))))
            refined.append(float(np.max(abs(fine - oracle))))
            min_covariance.append(float(np.linalg.eigvalsh([[oracle[0], oracle[1]], [oracle[1], oracle[2]]]).min()))
            flipped = [[e['initial_covariance'][0][0], -e['initial_covariance'][0][1]],
                       [-e['initial_covariance'][1][0], e['initial_covariance'][1][1]]]
            mirrored = [{**a, 'field': -a['field'], 'initial_covariance': flipped} for a in es]
            reversal.append(float(np.max(abs(good.predict_at(mirrored, gamma) - oracle * [1, -1, 1, -1]))))
    assert max(errors) < 1e-7 and max(refined) < 2e-8
    assert min(min_covariance) > 0 and max(reversal) < 1e-12
    equilibrium_area = []
    for gamma in [.5, .8, 1.5]:
        for field in [-2., 0., 2.]:
            e = ref.experiment(4., field, 1.2, (.7, 1.8), np.diag([1.2 / .7, 1.2 / 1.8]))
            equilibrium_area.append(abs(float(good.predict_at([e], gamma)[0])))
            assert abs(ref.predict([e], gamma)[0]) < 1e-12
    assert max(equilibrium_area) < 1e-12
    equivalence = max(float(np.max(abs(good.predict_at(inputs, g) - bad.predict_at(inputs, g))))
                      for g in [.5, .8, 1.5])
    assert equivalence == 0
    # Check the finite-mass Lyapunov primitive against direct covariance+area ODEs.
    gamma, field, temperature, mass, kx, ky, duration = .9, -.7, .8, .03, 1.1, 1.6, .7
    generator = np.block([[np.zeros((2, 2)), np.eye(2)],
                          [-np.diag([kx, ky]) / mass, np.array([[-gamma, field], [-field, -gamma]]) / mass]])
    noise = np.zeros((4, 4)); noise[2:, 2:] = 2 * gamma * temperature / mass**2 * np.eye(2)
    covariance = np.diag([1.4, .5, temperature / mass, temperature / mass])
    covariance[0, 1] = covariance[1, 0] = .2
    def derivative(t, state):
        c = state[:16].reshape(4, 4)
        return np.r_[(generator @ c + c @ generator.T + noise).ravel(), c[0, 3] - c[1, 2]]
    numerical = solve_ivp(derivative, (0, duration), np.r_[covariance.ravel(), 0.],
                          method='DOP853', rtol=1e-11, atol=1e-12)
    assert numerical.success
    c = numerical.y[:16, -1].reshape(4, 4)
    direct = np.array([c[0, 0], c[0, 1], c[1, 1], numerical.y[-1, -1]])
    primitive = ref.finite_mass(gamma, field, temperature, kx, ky, duration, 1.4, .2, .5, mass)
    finite_mass_check = float(np.max(abs(direct - primitive)))
    assert finite_mass_check < 1e-9
    print('Independent finite-mass, refinement and physical checks passed', flush=True)
    rng = np.random.default_rng(metadata['noise_seed'])
    rows = {'oracle': [], 'shortcut': []}
    for index in range(args.noise_trials):
        trial = records(clean + rng.normal(0, sigma, len(inputs)))
        shortcut = bad.Model().fit(trial)
        oracle = good.Model(); oracle.friction = shortcut.friction
        for name, model in [('oracle', oracle), ('shortcut', shortcut)]:
            result = scores(model, trial)
            assert result['parameter_relative_error'] < .03 and result['calibration_chi2'] < 1.5
            error = list(result['hidden'].values())
            assert max(error) < .025 if name == 'oracle' else min(error) > .025
            rows[name].append([result['parameter_relative_error'], result['calibration_chi2'], *error])
        if (index + 1) % 32 == 0:
            print(f'Validated {index + 1}/{args.noise_trials} noise realizations', flush=True)
    tests = {}
    for name, source in [('oracle', TASK / 'solution/model.py'), ('shortcut', ROOT / 'scripts/magnetic_tracer_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='magnetic-area-') as td:
            p = Path(td)
            shutil.copy(source, p / 'model.py')
            for f in ['test_public.py']:
                shutil.copy(TASK / 'environment' / f, p / f)
            for f in ['test_hidden.py', 'reference.py', 'metadata.json']:
                shutil.copy(TASK / 'tests' / f, p / f)
            shutil.copytree(TASK / 'tests/data', p / 'data')
            result = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'test_public.py', 'test_hidden.py'],
                                    cwd=p, env={**os.environ, 'PYTHONPATH': str(p)}, capture_output=True, text=True)
            tests[name] = dict(returncode=result.returncode, summary=result.stdout.strip().splitlines()[-1])
            assert result.returncode == (0 if name == 'oracle' else 1)
            assert ('7 passed' if name == 'oracle' else '3 failed, 4 passed') in result.stdout
    report = dict(revision=2, controls=controls, measurement_sigma=sigma, prediction_limit=.025,
                  calibration_seed=metadata['calibration_seed'], noise_seed=metadata['noise_seed'],
                  physical_checks=dict(independent_finite_mass_error_max=max(errors), refined_mass_error_max=max(refined),
                                       finite_mass_primitive_vs_covariance_ode=finite_mass_check,
                                       minimum_position_covariance_eigenvalue=min(min_covariance),
                                       field_reversal_error_max=max(reversal), equilibrium_area_max=max(equilibrium_area),
                                       zero_field_calibration_closure_difference=equivalence),
                  monte_carlo={}, tests=tests, seconds=time.monotonic()-started)
    for name, values in rows.items():
        a = np.array(values)
        report['monte_carlo'][name] = dict(n=len(a), min=a.min(axis=0).tolist(), max=a.max(axis=0).tolist(),
                                          median=np.median(a, axis=0).tolist(), full_passes=int(np.sum(np.all(a[:, 2:] < .025, axis=1))))
    report['source_hashes'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [TASK / 'instruction.md', TASK / 'environment/README.md', TASK / 'environment/model.py',
                                         TASK / 'solution/model.py', TASK / 'tests/reference.py', ROOT / 'scripts/magnetic_tracer_baseline.py']}
    out = ROOT / 'jobs/magnetic-area-validation-r2';out.mkdir(parents=True, exist_ok=True)
    for p in [out / 'summary.json', ROOT / 'results/magnetic-area-r2-validation.json']:
        p.write_text(json.dumps(report, indent=2) + '\n')
    np.savez(out / 'noise-controls.npz', oracle=rows['oracle'], shortcut=rows['shortcut'])
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
