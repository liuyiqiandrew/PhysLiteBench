"""Validate staged entropy revision 5 without Docker or model-agent calls."""
import argparse
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/entropy-anomaly'
REPO = ROOT.parents[1]
RESULTS = ROOT/'results'
NOISE_SEED = 9161055


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def error(actual, expected):
    return float(np.sqrt(np.mean((actual-expected)**2))/np.sqrt(np.mean(expected**2)))


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_controls():
    output = {}
    for label, source in [('oracle', TASK/'solution/model.py'),
                          ('shortcut', ROOT/'scripts/entropy_anomaly_baseline.py')]:
        folder = RESULTS/'entropy-anomaly-r5-controls'/label
        app = folder/'app'
        app.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, app/'model.py')
        shutil.copyfile(TASK/'environment/test_public.py', app/'test_public.py')
        (app/'data').mkdir(exist_ok=True)
        shutil.copyfile(TASK/'environment/data/calibration.json', app/'data/calibration.json')
        env = {**os.environ, 'PYTHONPATH': str(app), 'PYTHONDONTWRITEBYTECODE': '1',
               'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1', 'OPENBLAS_NUM_THREADS': '1',
               'METRICS_PATH': str(folder/'metrics.json')}
        start = time.monotonic()
        command = [sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                   str(app/'test_public.py'), str(TASK/'tests/test_hidden.py')]
        run = subprocess.run(command, cwd=app, env=env, text=True, capture_output=True, timeout=60)
        (folder/'pytest.txt').write_text(run.stdout+run.stderr)
        metrics = json.loads((folder/'metrics.json').read_text())
        assert metrics['calibration_chi2'] < 1.5 and metrics['parameter_relative_error'] < .03
        assert metrics['hidden']['equal_contact_anchor'] < .04
        for key, value in metrics['hidden'].items():
            if key != 'equal_contact_anchor':
                assert (value < .04) == (label == 'oracle')
        assert run.returncode == (0 if label == 'oracle' else 1), run.stdout+run.stderr
        output[label] = dict(exit_code=run.returncode, seconds=time.monotonic()-start,
                             output=run.stdout.strip(), metrics=metrics)
    return output


def finite_mass_uniform_formula(e, gamma, mass):
    drag = np.diag([gamma, gamma*e['drag_ratio']])
    friction_matrix = np.array([[gamma, -e['magnetic']], [e['magnetic'], gamma*e['drag_ratio']]])
    mean_velocity = np.linalg.solve(friction_matrix, [e['force_x'], e['force_y']])
    collision = np.kron(friction_matrix, np.eye(2))+np.kron(np.eye(2), friction_matrix)
    switch = mass*e['switch_rate']*np.eye(4)
    temperatures = e['temperature']*(1+e['bath_contrast']*np.array([1., -1.]))
    system = np.block([[collision+switch, -switch], [-switch, collision+switch]])
    rhs = np.r_[(drag*temperatures[0]).ravel(), (drag*temperatures[1]).ravel()]
    moments = np.linalg.solve(system, rhs).reshape(2, 2, 2)
    entropy = 0.
    for temperature, covariance in zip(temperatures, moments):
        entropy += np.sum(np.diag(drag)/mass*(np.diag(covariance)/temperature-.5))
        entropy += .5*mean_velocity@drag@mean_velocity/temperature
    return float(entropy)


def physics_checks(ref, oracle, shortcut, truths):
    inputs = [e for group in ref.hidden_inputs().values() for e in group]
    truth = np.concatenate(list(truths.values()))
    gamma = ref.TRUE_PARAMETER
    start = time.monotonic()
    half_mass = ref.predict(inputs, gamma, epsilon=.003)
    refined = ref.predict(inputs, gamma, degree=16, spatial=49)
    direct = oracle.coefficients(inputs, gamma)
    errors = dict(oracle_reference_max_abs=float(np.max(abs(direct-truth))),
                  mass_refinement_max_abs=float(np.max(abs(half_mass-truth))),
                  space_velocity_refinement_max_abs=float(np.max(abs(refined-truth))))
    assert max(errors.values()) < 2e-6, errors
    print('entropy reference and mass/basis refinements pass', flush=True)

    identity = {key: 0. for key in ['state_normalization', 'current_balance', 'contact_term',
                                  'uniform_bath', 'zero_force', 'field_reflection',
                                  'label_reversal']}
    minimum_density = np.inf
    minimum_entropy = np.inf
    corners = 0
    for friction, ratio, field, temp, contrast, delta, rate, force_x, force_y in itertools.product(
            [.7, 1.6], [.25, 8.], [-5., 5.], [.8, 1.4], [0., .65], [0., .35], [.2, 2.], [-1.5, 1.5], [-1.5, 1.5]):
        e = ref.experiment(force_x, force_y, ratio, field, temp, contrast, 3, delta, rate)
        temperatures, gradients, density, jx, jy, dx = oracle.positional_state(e, friction)
        _, derivative = oracle.spatial_grid(density.shape[1])
        derivative = derivative*e['wavenumber']
        minimum_density = min(minimum_density, float(density.min()))
        identity['state_normalization'] = max(identity['state_normalization'], float(np.max(abs(density.sum(axis=1)*dx-.5))))
        identity['current_balance'] = max(identity['current_balance'], float(np.max(abs((derivative@jx.T).T-rate*(density[::-1]-density)))))
        actual = oracle.coefficients([e], friction)[0]
        alternative = shortcut.coefficients([e], friction)[0]
        minimum_entropy = min(minimum_entropy, actual)
        exact_contact = 2*rate*delta**2/(1-delta**2)
        identity['contact_term'] = max(identity['contact_term'], abs(actual-alternative-exact_contact))
        reflected = dict(e, magnetic=-field, force_y=-force_y)
        identity['field_reflection'] = max(identity['field_reflection'], abs(actual-oracle.coefficients([reflected], friction)[0]))
        reversed_label = dict(e, bath_contrast=-delta)
        identity['label_reversal'] = max(identity['label_reversal'], abs(actual-oracle.coefficients([reversed_label], friction)[0]))
        if contrast == 0:
            f = np.array([force_x, force_y])
            matrix = np.array([[friction, -field], [field, friction*ratio]])
            exact = f@np.linalg.solve(matrix, f)/(temp*(1-delta**2))+exact_contact
            identity['uniform_bath'] = max(identity['uniform_bath'], abs(actual-exact))
        corners += 1
    assert minimum_density > 0 and minimum_entropy >= 0
    assert max(identity.values()) < 2e-7, identity

    for ratio, field, delta, rate in itertools.product([.25, 1., 8.], [-5., 0., 5.], [0., .35], [.2, 2.]):
        e = ref.experiment(0., 0., ratio, field, 1.4, .65, 3, delta, rate)
        expected = oracle.heat_coefficient(gamma, ratio, field)*e['temperature']*e['wavenumber']**2*(1-np.sqrt(1-e['contrast']**2))
        expected += 2*rate*delta**2/(1-delta**2)
        identity['zero_force'] = max(identity['zero_force'], abs(oracle.coefficients([e], gamma)[0]-expected))
    assert identity['zero_force'] < 2e-7
    print('entropy 512 domain corners and analytical identities pass', flush=True)

    direct_heat_errors = []
    scalar_rate_errors = []
    balance_errors = []
    covariance_minimum = np.inf
    for e in [ref.experiment(0., 0., .4, 2.2, 1., 0., 1, .35, .8),
              ref.experiment(.2, -.15, 2., -1.8, 1.2, 0., 2, .3, 1.2)]:
        mass = .001
        physical = ref.finite_mass_entropy(e, gamma, mass)
        direct_heat_errors.append(abs(physical-finite_mass_uniform_formula(e, gamma, mass)))
    for ratio, delta, rate in [(1., .35, .8), (.25, .3, 2.), (8., .2, .2)]:
        e = ref.experiment(0., 0., ratio, 0., 1., 0., 1, delta, rate)
        mass = .001
        expected = sum(drag*rate*delta**2/((drag+mass*rate)*(1-delta**2))
                       for drag in [gamma, gamma*ratio])
        scalar_rate_errors.append(abs(ref.finite_mass_entropy(e, gamma, mass)-expected))
    for e in [inputs[0], inputs[7], inputs[9]]:
        mass = .001*(gamma*min(1., e['drag_ratio']))**2/(e['temperature']*e['wavenumber']**2)
        phase, temperatures, states, basis_temp = ref.kinetic_state(e, gamma, mass, degree=16, spatial=49)
        gradients = np.array([-e['temperature']*(1+sign*e['bath_contrast'])*e['contrast']*e['wavenumber']*np.sin(phase)
                              for sign in [1, -1]])
        energies = [basis_temp*(c[0, 0]+np.sqrt(2)/2*(c[2, 0]+c[0, 2])) for c in states]
        moment_rate = 0.
        for s, (temperature, gradient, c) in enumerate(zip(temperatures, gradients, states)):
            jx = np.sqrt(basis_temp/mass)*c[1, 0]
            jy = np.sqrt(basis_temp/mass)*c[0, 1]
            flux = basis_temp**1.5/(2*np.sqrt(mass))*(4*c[1, 0]+np.sqrt(6)*c[3, 0]+np.sqrt(2)*c[1, 2])
            moment_rate += 2*np.pi*np.mean((e['force_x']*jx+e['force_y']*jy)/temperature-
                flux*gradient/temperature**2+e['switch_rate']*(energies[1-s]-energies[s])/temperature)
            covariance = np.empty((len(phase), 2, 2))
            covariance[:, 0, 0] = 1+np.sqrt(2)*c[2, 0]/c[0, 0]-(c[1, 0]/c[0, 0])**2
            covariance[:, 1, 1] = 1+np.sqrt(2)*c[0, 2]/c[0, 0]-(c[0, 1]/c[0, 0])**2
            covariance[:, 0, 1] = covariance[:, 1, 0] = c[1, 1]/c[0, 0]-c[1, 0]*c[0, 1]/c[0, 0]**2
            covariance_minimum = min(covariance_minimum, float(np.linalg.eigvalsh(covariance).min()))
        balance_errors.append(abs(moment_rate-ref.finite_mass_entropy(e, gamma, mass, degree=16, spatial=49)))
    assert max(direct_heat_errors+balance_errors+scalar_rate_errors) < 2e-7 and covariance_minimum > 0

    corner = ref.experiment(.15, -.18, .25, 4.5, 1.4, .65, 3, .35, 2.)
    corner_errors = []
    corner_refinement_errors = []
    for friction in [.7, 1.6]:
        cold = ref.predict([corner], friction)[0]
        corner_errors.append(abs(cold-oracle.coefficients([corner], friction)[0]))
        corner_refinement_errors.append(abs(cold-ref.predict([corner], friction, degree=16, spatial=49)[0]))
    assert max(corner_errors) < 2e-6
    assert max(corner_refinement_errors) < 2e-7
    return dict(reference_errors=errors, identities=identity, domain_corner_count=corners,
                minimum_position_density=minimum_density, minimum_entropy=minimum_entropy,
                uniform_finite_mass_reference_error=max(direct_heat_errors),
                uniform_zero_field_scalar_formula_error=max(scalar_rate_errors),
                finite_mass_energy_balance_error=max(balance_errors),
                minimum_conditional_scaled_velocity_covariance=covariance_minimum,
                allowed_corner_reference_error=max(corner_errors),
                allowed_corner_refinement_error=max(corner_refinement_errors), seconds=time.monotonic()-start)


def validate(noise_trials):
    start = time.monotonic()
    ref = module(TASK/'tests/reference.py', 'entropy_r5_reference')
    oracle = module(TASK/'solution/model.py', 'entropy_r5_oracle')
    shortcut = module(ROOT/'scripts/entropy_anomaly_baseline.py', 'entropy_r5_shortcut')
    previous = module(REPO/'tasks/entropy-anomaly/solution/model.py', 'entropy_previous')
    records = json.loads((TASK/'environment/data/calibration.json').read_text())
    private = json.loads((TASK/'tests/data/calibration.json').read_text())
    original = json.loads((REPO/'tasks/entropy-anomaly/environment/data/calibration.json').read_text())
    assert records == private and len(records) == len(original) == 192
    for new, old in zip(records, original):
        assert new['value'] == old['value'] and new['sigma'] == old['sigma']
        assert {k: v for k, v in new['input'].items() if k not in ['bath_contrast', 'switch_rate']} == old['input']
        assert new['input']['bath_contrast'] == 0
    inputs = [r['input'] for r in records]
    true = ref.TRUE_PARAMETER
    noiseless = previous.coefficients(inputs, true)
    sigma = np.array([r['sigma'] for r in records])
    coefficient = oracle.coefficients(inputs)
    equivalence = {}
    for gamma in [.7, 1.1, 1.6]:
        old = previous.coefficients(inputs, gamma)
        for label, mod in [('oracle', oracle), ('shortcut', shortcut)]:
            equivalence[label+str(gamma)] = float(np.max(abs(old-mod.coefficients(inputs, gamma))))
    assert max(equivalence.values()) < 2e-7
    sample_indices = [0, 7, 19, 31, 47, 63, 79, 95]
    cal_ref = ref.predict([inputs[i] for i in sample_indices], true)
    calibration_reference_error = float(np.max(abs(cal_ref-noiseless[sample_indices])))
    assert calibration_reference_error < 2e-6
    hidden = ref.hidden_inputs()
    truths = {key: ref.predict(es, true) for key, es in hidden.items()}
    controls = {}
    for label, mod in [('oracle', oracle), ('shortcut', shortcut)]:
        model = mod.Model().fit(records)
        residual = (model.predict(inputs)-np.array([r['value'] for r in records]))/sigma
        metrics = dict(parameter=model.friction, parameter_relative_error=abs(model.friction/true-1),
                       calibration_chi2=float(residual@residual/(len(records)-1)),
                       hidden={key: error(model.predict(es), truths[key]) for key, es in hidden.items()})
        assert metrics['parameter_relative_error'] < .03 and metrics['calibration_chi2'] < 1.5
        assert metrics['hidden']['equal_contact_anchor'] < .04
        for key, value in metrics['hidden'].items():
            if key != 'equal_contact_anchor':
                assert (value < .04) == (label == 'oracle')
        controls[label] = metrics
    print('entropy fitted controls pass intended gates', flush=True)

    rng = np.random.default_rng(NOISE_SEED)
    noise = {label: dict(max_parameter_error=0., max_chi2=0., hidden_min=1e10,
                         hidden_max=0., anchor_max=0., successful_trials=0)
             for label in ['oracle', 'shortcut']}
    fitted = []
    for i in range(noise_trials):
        values = noiseless+rng.normal(0, sigma)
        inverse = np.sum(coefficient*values/sigma**2)/np.sum(coefficient**2/sigma**2)
        gamma = float(1/inverse)
        chi2 = float(np.sum(((coefficient/gamma-values)/sigma)**2)/(len(values)-1))
        fitted.append(gamma)
        sample = [dict(input=e, value=float(v), sigma=float(s)) for e, v, s in zip(inputs, values, sigma)]
        for label, mod in [('oracle', oracle), ('shortcut', shortcut)]:
            model = mod.Model().fit(sample)
            assert abs(model.friction-gamma) < 1e-12
            if i in [0, noise_trials-1]:
                assert np.max(abs(model.predict(inputs)-coefficient/gamma)) < 2e-7
            scores = {key: error(model.predict(es), truths[key]) for key, es in hidden.items()}
            diagnostic = [v for key, v in scores.items() if key != 'equal_contact_anchor']
            assert abs(gamma/true-1) < .03 and chi2 < 1.5 and scores['equal_contact_anchor'] < .04
            assert (max(diagnostic) < .04) if label == 'oracle' else (min(diagnostic) > .04)
            row = noise[label]
            row['max_parameter_error'] = max(row['max_parameter_error'], abs(gamma/true-1))
            row['max_chi2'] = max(row['max_chi2'], chi2)
            row['hidden_min'] = min(row['hidden_min'], min(diagnostic))
            row['hidden_max'] = max(row['hidden_max'], max(diagnostic))
            row['anchor_max'] = max(row['anchor_max'], scores['equal_contact_anchor'])
            row['successful_trials'] += 1
    print('entropy all 256 noisy calibration/hidden control draws pass', flush=True)
    report = dict(revision=5, status='scientific_and_local_controls_pass_model_evaluation_pending',
                  calibration_records=len(records), old_values_and_sigma_preserved=True,
                  calibration_copies_equal=True, original_calibration_sha256=fingerprint(REPO/'tasks/entropy-anomaly/environment/data/calibration.json'),
                  calibration_equivalence=equivalence, calibration_finite_mass_reference_error=calibration_reference_error,
                  controls=controls, noise_trials=noise_trials, noise_seed=NOISE_SEED, noise=noise,
                  fitted_parameter_range=[min(fitted), max(fitted)],
                  independent_checks=physics_checks(ref, oracle, shortcut, truths),
                  local_controls=local_controls())
    report['seconds'] = time.monotonic()-start
    files = [p for p in TASK.rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    files += [ROOT/'scripts/entropy_anomaly_baseline.py', Path(__file__).resolve()]
    report['source_sha256'] = {str(p.relative_to(ROOT)): fingerprint(p) for p in files}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--noise-trials', type=int, default=256)
    parser.add_argument('--output', type=Path, default=RESULTS/'entropy-anomaly-r5-validation.json')
    args = parser.parse_args()
    report = validate(args.noise_trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('entropy r5 validation PASS', report['seconds'], flush=True)


if __name__ == '__main__':
    main()
