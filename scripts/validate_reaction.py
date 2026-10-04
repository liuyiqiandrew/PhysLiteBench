"""Validate reaction-diffusion revision 7 without changing calibration data."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags, kron, csr_matrix

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/reaction-diffusion'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = load('reference', TASK / 'tests/reference.py')
Oracle = load('oracle', TASK / 'solution/model.py').TransportModel
Shortcut = load('shortcut', ROOT / 'scripts/reaction_baseline.py').TransportModel


def check(model, data, physical):
    result = ref.metrics(model, data)
    assert result['relative_diffusivity_error'] < .05
    assert result['calibration_reduced_chi2'] < 1.5
    assert min(result['minimum_hidden_concentration']) > -1e-8
    assert max(result['maximum_mean_concentration_error']) < 1e-8
    if physical:
        assert max(result['hidden_nrmse']) < .01
    else:
        assert min(result['hidden_nrmse']) > .05
    return result


def transport_matrix(mean):
    r, s = mean / (mean[0] + 2 * mean[1])
    return np.array([[1 + r / 3, -2 * r / 3],
                     [4 * s / 3, 4 * (1 - 2 * s / 3)]])


def linear_reference(t, x, initial, diffusivity):
    """Independent method-of-lines solve of the completed closure."""
    count = len(x)
    dx = x[1] - x[0]
    laplacian = diags([np.ones(count - 1), -2 * np.ones(count),
                      np.ones(count - 1)], [-1, 0, 1], format='lil') / dx**2
    laplacian[0, 1] *= 2
    laplacian[-1, -2] *= 2
    mean = np.trapezoid(initial, x, axis=0) / (x[-1] - x[0])
    generator = diffusivity * kron(laplacian.tocsr(), csr_matrix(transport_matrix(mean)), format='csr')
    result = solve_ivp(lambda time, state: generator @ state, (0, t[-1]),
                       initial.ravel(), t_eval=t, method='BDF', jac=generator,
                       rtol=2e-10, atol=2e-12)
    assert result.success
    return result.y.T.reshape(len(t), count, 2)


def structure_checks():
    symmetry_error, minimum_dissipation = 0., np.inf
    for a in np.geomspace(.01, 100, 19):
        for b in np.geomspace(.01, 100, 19):
            c = a + b
            hessian = np.array([[1 / a + 1 / c, 1 / c],
                                [1 / c, 1 / b + 1 / c]])
            product = hessian @ transport_matrix(np.array([a, b]))
            symmetry_error = max(symmetry_error, float(np.max(abs(product - product.T))))
            minimum_dissipation = min(minimum_dissipation, float(np.linalg.eigvalsh(product).min()))
    assert symmetry_error < 1e-12 and minimum_dissipation > 0
    model = Shortcut()
    model.diffusivity = ref.D
    x = np.linspace(0, ref.L, 65)
    t = np.linspace(0, 250, 31)
    minimum, mass_error, energy_increase = np.inf, 0., 0.
    # Finite-amplitude profiles surrounding the scored cases, including both
    # species perturbed and the pure-salt endpoints. This is not a global
    # positivity claim for arbitrary finite-amplitude cross-diffusion data.
    profiles = []
    z = np.pi * x / ref.L
    for a, b in [(1., .8), (.6, 1.), (1., .6), (.2, 2.), (2., .2)]:
        for mode in [1, 2, 3]:
            for sign in [-1, 0, 1]:
                profiles.append(np.column_stack((a * (1 + .7 * np.cos(mode * z)),
                                                  b * (1 + sign * .7 * np.cos(mode * z)))))
    for species in [0, 1]:
        initial = np.zeros((len(x), 2))
        initial[:, species] = 1 + .7 * np.cos(z)
        profiles.append(initial)
    for initial in profiles:
        prediction = model.predict(t, x, initial)
        mean = np.trapezoid(initial, x, axis=0) / ref.L
        minimum = min(minimum, float(prediction.min()))
        mass_error = max(mass_error, float(np.max(abs(np.trapezoid(prediction, x, axis=1) / ref.L - mean))))
        if np.all(mean > 0):
            c = mean.sum()
            hessian = np.diag(1 / mean) + np.ones((2, 2)) / c
            delta = prediction - mean
            density = .5 * np.einsum('txi,ij,txj->tx', delta, hessian, delta)
            energy = np.trapezoid(density, x, axis=1)
            energy_increase = max(energy_increase, float(np.max(np.diff(energy))))
    assert minimum > -1e-10 and mass_error < 1e-10 and energy_increase < 1e-12
    return {'backgrounds': 361, 'max_entropy_symmetrizer_asymmetry': symmetry_error,
            'min_entropy_dissipation_eigenvalue': minimum_dissipation,
            'profile_checks': len(profiles), 'minimum_concentration': minimum,
            'maximum_mean_concentration_error': mass_error,
            'maximum_quadratic_entropy_increase': energy_increase}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-trials', type=int, default=256)
    args = parser.parse_args()
    out = ROOT / 'jobs/reaction-validation-r7'
    out.mkdir(parents=True, exist_ok=True)
    data = ref.load_data()
    controls = {name: check(cls().fit(data), data, name == 'oracle')
                for name, cls in [('oracle', Oracle), ('shortcut', Shortcut)]}
    print('Fixed-data controls passed', flush=True)
    oracle, shortcut = Oracle(), Shortcut()
    oracle.diffusivity = shortcut.diffusivity = ref.D
    truth = np.array([oracle.predict(data['t'], data['x'], initial) for initial in data['initial']])
    pure_error = float(np.max(abs(truth - np.array([shortcut.predict(data['t'], data['x'], initial)
                                                   for initial in data['initial']]))))
    assert pure_error < 1e-7
    numerical, convergence, linear_error = [], [], []
    for t, x, initial, expected in ref.hidden_cases():
        mean = np.trapezoid(initial, x, axis=0) / ref.L
        numerical.append((np.linalg.norm(oracle.predict(t, x, initial) - expected, axis=(0, 1))
                          / np.linalg.norm(expected - mean, axis=(0, 1))).tolist())
        fine = ref.trajectory(t, x, initial, refinement=8)
        convergence.append((np.linalg.norm(expected - fine, axis=(0, 1))
                            / np.linalg.norm(fine - mean, axis=(0, 1))).tolist())
        linear_error.append(float(np.max(abs(shortcut.predict(t, x, initial)
                                            - linear_reference(t, x, initial, ref.D)))))
    assert np.max(numerical) < .005 and np.max(convergence) < .0001
    assert max(linear_error) < 1e-7
    structure = structure_checks()
    print('Independent references and structural checks passed', flush=True)
    rng = np.random.default_rng(1729)
    rows = {'oracle': [], 'shortcut': []}
    for index in range(args.noise_trials):
        trial = {**data, 'concentration': truth + rng.normal(size=truth.shape) * data['sigma']}
        shortcut = Shortcut().fit(trial)
        oracle = Oracle()
        oracle.diffusivity = shortcut.diffusivity
        for name, model in [('oracle', oracle), ('shortcut', shortcut)]:
            result = check(model, trial, name == 'oracle')
            rows[name].append([result['relative_diffusivity_error'], result['calibration_reduced_chi2'],
                               *result['hidden_nrmse']])
        if (index + 1) % 32 == 0:
            print(f'Validated {index + 1}/{args.noise_trials} noise realizations', flush=True)
    summary = {'revision': 7, 'seed': 1729, 'controls': controls,
               'independent_reference_species_nrmse': numerical,
               'reference_refinement4_vs8_species_nrmse': convergence,
               'independent_linear_reference_max_abs_error': linear_error,
               'noiseless_pure_salt_oracle_shortcut_max_abs_error': pure_error,
               'structure': structure, 'monte_carlo': {},
               'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [TASK / 'instruction.md', TASK / 'environment/model.py',
                                           TASK / 'solution/model.py', TASK / 'tests/reference.py',
                                           ROOT / 'scripts/reaction_baseline.py']}}
    for name, values in rows.items():
        a = np.asarray(values)
        summary['monte_carlo'][name] = {'n': len(a), 'min': a.min(axis=0).tolist(),
                                      'median': np.median(a, axis=0).tolist(), 'max': a.max(axis=0).tolist(),
                                      'full_passes': int(np.sum(np.all(a[:, 2:] < .05, axis=1)))}
    for path in [out / 'summary.json', ROOT / 'results/reaction-r7-validation.json']:
        path.write_text(json.dumps(summary, indent=2) + '\n')
    np.savez(out / 'noise-controls.npz', oracle=rows['oracle'], shortcut=rows['shortcut'])
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
