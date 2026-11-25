"""Check Thomson heating against independent heat-flux and electrical-work balance."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/thermoelectric-rod'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def chi2(model, data):
    residuals = []
    for i, initial in enumerate(data['initial']):
        output = model.predict(data['t'], data['x'], initial, data['current'][i], data['boundary'][i])
        for field in ['temperature', 'voltage']:
            residuals.extend(((output[field]-data[field][i])/data['sigma_'+field][i]).ravel())
    return float(np.sum(np.square(residuals))/(len(residuals)-1))


def noiseless(data, ref):
    runs = [ref.predict(data['t'], data['x'], initial, data['current'][i],
                        data['boundary'][i], ref.TRUE_CONDUCTIVITY, refinement=8)
            for i, initial in enumerate(data['initial'])]
    return {field: np.array([r[field] for r in runs]) for field in ['temperature', 'voltage']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--noise-trials', type=int, default=256)
    args = parser.parse_args()
    oracle = module('electric_oracle', TASK/'solution/model.py')
    shortcut = module('electric_shortcut', ROOT/'scripts/thermoelectric_rod_baseline.py')
    ref = module('electric_reference', TASK/'tests/reference.py')
    if args.generate:
        x = np.linspace(0, .01, 31)
        t = np.array([1., 2., 4., 7., 10., 15., 22., 32., 45., 65.])
        boundary = np.array([[290., 330.], [300., 300.], [310., 280.]])
        profiles = [np.linspace(*b, len(x))+a*np.sin(np.pi*x/.01)
                    for b, a in zip(boundary, [10., -10., 12.])]
        data = {'x': x, 't': t, 'initial': np.array(profiles), 'current': np.zeros(3), 'boundary': boundary}
        mean = noiseless(data, ref)
        rng = np.random.default_rng(9401)
        for field, sigma in [('temperature', .04), ('voltage', 2e-5)]:
            data['sigma_'+field] = np.full_like(mean[field], sigma)
            data[field] = mean[field]+rng.normal(size=mean[field].shape)*sigma
        path = TASK/'environment/data/calibration.npz'
        np.savez(path, **data)
        shutil.copyfile(path, TASK/'tests/data/calibration.npz')
    assert (TASK/'environment/data/calibration.npz').read_bytes() == (TASK/'tests/data/calibration.npz').read_bytes()
    data = dict(np.load(TASK/'environment/data/calibration.npz'))
    result = {'revision': 1, 'calibration_seed': 9401, 'noise_seed': 19401,
              'noise_trials': args.noise_trials, 'controls': {}}
    for name, implementation in [('oracle', oracle), ('shortcut', shortcut)]:
        measured = ref.metrics(implementation.Model().fit(data), data)
        result['controls'][name] = measured
        assert measured['calibration_chi2'] < 1.5 and measured['parameter_relative_error'] < .03
        assert all((v < .04) == (name == 'oracle') for v in measured['hidden'].values())
        assert max(measured['voltage_absolute_error']) < 1e-5
    a, b = oracle.Model(), shortcut.Model()
    a.conductivity = b.conductivity = ref.TRUE_CONDUCTIVITY
    difference = 0.
    for i, initial in enumerate(data['initial']):
        settings = (data['t'], data['x'], initial, data['current'][i], data['boundary'][i])
        for field in ['temperature', 'voltage']:
            difference = max(difference, float(np.max(abs(a.predict(*settings)[field]-b.predict(*settings)[field]))))
    assert difference < 1e-12
    mean = noiseless(data, ref)
    rng = np.random.default_rng(19401)
    fitted, chi = [], []
    for i in range(args.noise_trials):
        noisy = dict(data)
        for field in ['temperature', 'voltage']:
            noisy[field] = mean[field]+rng.normal(size=mean[field].shape)*data['sigma_'+field]
        model = oracle.Model().fit(noisy)
        fitted.append(model.conductivity)
        chi.append(chi2(model, noisy))
    assert max(chi) < 1.5 and max(abs(np.array(fitted)/ref.TRUE_CONDUCTIVITY-1)) < .03
    separation = {'oracle': [], 'shortcut': []}
    convergence, true_error = [], []
    for settings in ref.hidden_inputs():
        truth = ref.predict(*settings, ref.TRUE_CONDUCTIVITY, refinement=4)
        fine = ref.predict(*settings, ref.TRUE_CONDUCTIVITY, refinement=8)
        convergence.append(float(np.max(abs(truth['temperature']-fine['temperature']))))
        true_error.append(float(np.max(abs(a.predict(*settings)['temperature']-truth['temperature']))))
        t, x, initial, current, boundary = settings
        scale = np.sqrt(np.mean((truth['temperature']-np.linspace(*boundary, len(x)))**2))
        for name, implementation in [('oracle', oracle), ('shortcut', shortcut)]:
            for k in [min(fitted), max(fitted)]:
                model = implementation.Model()
                model.conductivity = k
                predicted = model.predict(*settings)['temperature']
                error = float(np.sqrt(np.mean((predicted-truth['temperature'])**2))/scale)
                separation[name].append(error)
                assert (error < .04) == (name == 'oracle')
    assert max(convergence) < .002 and max(true_error) < .02
    # Reversing the rod and current reverses the temperature profile.
    settings = ref.hidden_inputs()[0]
    t, x, initial, current, boundary = settings
    direct = a.predict(*settings)['temperature']
    reverse = a.predict(t, x, initial[::-1], -current, boundary[::-1])['temperature'][:, ::-1]
    symmetry = float(np.max(abs(direct-reverse)))
    assert symmetry < 1e-6
    result['physical_checks'] = {'zero_current_equivalence': difference,
                                 'reference_mesh_max_difference_kelvin': max(convergence),
                                 'independent_energy_form_max_difference_kelvin': max(true_error),
                                 'current_spatial_reversal_error_kelvin': symmetry}
    result['noise'] = {'calibration_chi2_max': max(chi), 'calibration_pass_fraction': 1.,
                       'parameter_min': min(fitted), 'parameter_max': max(fitted),
                       'parameter_relative_error_max': float(max(abs(np.array(fitted)/ref.TRUE_CONDUCTIVITY-1)))}
    result['hidden_parameter_extrema_check'] = {k: {'min': min(v), 'max': max(v)} for k, v in separation.items()}
    output = ROOT/'jobs/thermoelectric-rod-validation'
    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
