"""Validate unsteady-sphere revision 1; regenerate calibration only with --generate."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['unsteady-sphere']


def module(path):
    spec = importlib.util.spec_from_file_location('candidate_'+str(abs(hash(str(path)))), path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def scale_for(name, experiments, truth):
    return np.full_like(truth, np.sqrt(np.mean(truth**2)))


def calibration_metrics(model, records, reference):
    values = np.array([r['value'] for r in records])
    sigma = np.array([r['sigma'] for r in records])
    pred = model.predict([r['input'] for r in records])
    return dict(parameter=float(getattr(model, reference.PARAMETER)),
                parameter_relative_error=float(abs(getattr(model, reference.PARAMETER)/reference.TRUE_PARAMETER-1)),
                calibration_chi2=float(np.sum(((pred-values)/sigma)**2)/(len(records)-1)))


def validate(name, generate, noise_trials):
    task = ROOT/'tasks'/name
    ref = module(task/'tests/reference.py')
    oracle_class = module(task/'solution/model.py').Model
    baseline_class = module(ROOT/'scripts'/(name.replace('-', '_')+'_baseline.py')).Model
    meta = json.loads((task/'tests/metadata.json').read_text())
    inputs = ref.calibration_inputs()*2
    noiseless = ref.predict(inputs, ref.TRUE_PARAMETER)
    sigma = .006*max(np.max(np.abs(noiseless)), 1e-10)
    seed = 9325
    if generate:
        noise = np.random.default_rng(seed).normal(0, sigma, len(inputs))
        records = [dict(input=e, value=float(v), sigma=float(sigma)) for e, v in zip(inputs, noiseless+noise)]
        for folder in ['environment', 'tests']:
            path = task/folder/'data/calibration.json'
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(records, indent=2)+'\n')
    records = json.loads((task/'environment/data/calibration.json').read_text())
    assert records == json.loads((task/'tests/data/calibration.json').read_text())
    report = {'revision':1, 'seed':seed, 'noise_trials':noise_trials, 'controls':{}, 'noise':{}}
    hidden = ref.hidden_inputs()
    truths = {key: ref.predict(es, ref.TRUE_PARAMETER) for key, es in hidden.items()}
    for label, cls in [('oracle', oracle_class), ('shortcut', baseline_class)]:
        exact = cls()
        setattr(exact, ref.PARAMETER, ref.TRUE_PARAMETER)
        calibration_delta = float(np.max(np.abs(exact.predict(inputs)-noiseless)))
        assert calibration_delta < 1e-6*max(1., np.max(np.abs(noiseless)))
        model = cls().fit(records)
        metrics = calibration_metrics(model, records, ref)
        assert metrics['calibration_chi2'] < 1.5 and metrics['parameter_relative_error'] < .03
        metrics['calibration_reference_max_error'] = calibration_delta
        metrics['hidden'] = {}
        for key, experiments in hidden.items():
            truth = truths[key]
            score = float(np.sqrt(np.mean(((model.predict(experiments)-truth)/scale_for(name, experiments, truth))**2)))
            metrics['hidden'][key] = score
        if label == 'oracle':
            assert max(metrics['hidden'].values()) < meta['prediction_limit'], metrics
        else:
            assert min(metrics['hidden'].values()) > meta['prediction_limit'], metrics
        report['controls'][label] = metrics

    # Both controls have exactly the same calibration predictor. Fit each noise
    # realization once, then verify the other fit on the first and last cases.
    rng = np.random.default_rng(seed+10000)
    fitted, chi2s = [], []
    for k in range(noise_trials):
        noisy = noiseless+rng.normal(0, sigma, len(inputs))
        sample = [dict(input=e, value=float(v), sigma=float(sigma)) for e, v in zip(inputs, noisy)]
        model = oracle_class().fit(sample)
        metrics = calibration_metrics(model, sample, ref)
        fitted.append(metrics['parameter'])
        chi2s.append(metrics['calibration_chi2'])
        assert metrics['parameter_relative_error'] < .03, metrics
        if k in [0, noise_trials-1]:
            other = baseline_class().fit(sample)
            assert abs(getattr(other, ref.PARAMETER)/metrics['parameter']-1) < 1e-5
    report['noise'] = {'chi2_max':max(chi2s), 'chi2_pass_fraction':float(np.mean(np.array(chi2s)<1.5)),
                       'parameter_min':min(fitted), 'parameter_max':max(fitted),
                       'parameter_relative_error_max':float(np.max(np.abs(np.array(fitted)/ref.TRUE_PARAMETER-1)))}
    assert report['noise']['chi2_pass_fraction'] >= .99
    # Check hidden predictions at the observed fitted-parameter extrema. This is
    # explicitly a sensitivity check, not a claim of exhaustive hidden MC trials.
    sensitivity = {}
    for label, cls in [('oracle', oracle_class), ('shortcut', baseline_class)]:
        scores = []
        for p in [min(fitted), max(fitted)]:
            model = cls()
            setattr(model, ref.PARAMETER, p)
            for key, es in hidden.items():
                score = float(np.sqrt(np.mean(((model.predict(es)-truths[key])/scale_for(name, es, truths[key]))**2)))
                scores.append(score)
        sensitivity[label] = {'min':min(scores), 'max':max(scores)}
    assert sensitivity['oracle']['max'] < meta['prediction_limit']
    assert sensitivity['shortcut']['min'] > meta['prediction_limit']
    report['hidden_parameter_extrema_check'] = sensitivity

    report['independent_checks'] = independent_checks(ref, oracle_class, baseline_class)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tasks', nargs='*', choices=NAMES)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--noise-trials', type=int, default=256)
    parser.add_argument('--output', type=Path, default=ROOT/'jobs/unsteady-sphere-validation/summary.json')
    args = parser.parse_args()
    if args.noise_trials < 1:
        parser.error('noise-trials must be positive')
    output = {}
    for name in args.tasks or NAMES:
        start = time.monotonic()
        output[name] = validate(name, args.generate, args.noise_trials)
        output[name]['seconds'] = time.monotonic()-start
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2)+'\n')
        print(name, 'PASS', round(output[name]['seconds'], 2), flush=True)


def independent_checks(ref, oracle_class, baseline_class):
    from scipy.integrate import quad
    model = oracle_class(); model.viscosity = ref.TRUE_PARAMETER
    baseline = baseline_class(); baseline.viscosity = model.viscosity
    all_es = [e for es in ref.hidden_inputs().values() for e in es]
    force_error = 0.
    minimum_real_impedance = float('inf')
    for eta in [.004,.008,.016]:
        for omega in [5.,20.,120.]:
            wave = np.sqrt(-1j*omega*1000/eta)
            exact = 6*np.pi*eta*.001*(1+.001*wave+(.001*wave)**2/9)
            traction = -ref.fluid_force_per_velocity(omega,eta)
            force_error = max(force_error,float(abs(traction/exact-1)))
            minimum_real_impedance = min(minimum_real_impedance,float(traction.real))
    assert force_error < 1e-12 and minimum_real_impedance>0
    # Numerically transform the causal t^-1/2 Basset kernel, including its phase.
    kernel_error = 0.
    eta,rho,a = model.viscosity,1000.,.001
    for omega in [5.,20.,120.]:
        real = quad(lambda t:np.cos(omega*t)/np.sqrt(t),0.,1.,epsabs=1e-11,epsrel=1e-11)[0]
        real += quad(lambda t:1/np.sqrt(t),1.,np.inf,weight='cos',wvar=omega,epsabs=1e-11,limit=200)[0]
        imaginary = quad(lambda t:np.sin(omega*t)/np.sqrt(t),0.,1.,epsabs=1e-11,epsrel=1e-11)[0]
        imaginary += quad(lambda t:1/np.sqrt(t),1.,np.inf,weight='sin',wvar=omega,epsabs=1e-11,limit=200)[0]
        memory = 6*a*a*np.sqrt(np.pi*eta*rho)*(-1j*omega)*(real+1j*imaginary)
        target = 6*np.pi*eta*a*a*np.sqrt(-1j*omega*rho/eta)
        kernel_error = max(kernel_error,float(abs(memory/target-1)))
    assert kernel_error < 1e-8
    # Kinetic energy of the irrotational dipole flow defines the added mass.
    radial = quad(lambda q:q**-4,1.,np.inf,epsabs=1e-12)[0]
    angular = quad(lambda z:z*z+(1-z*z)/4,-1.,1.,epsabs=1e-12)[0]
    added_mass = rho*2*np.pi*a**3*radial*angular
    half_displaced = rho*2*np.pi*a**3/3
    added_mass_error = float(abs(added_mass/half_displaced-1))
    assert added_mass_error < 1e-12
    globals_ = model.predict.__func__.__globals__
    density = globals_['FLUID_DENSITY']
    globals_['FLUID_DENSITY'] = 0.
    zero_density_error = float(np.max(abs(model.predict(all_es)-baseline.predict(all_es))))
    globals_['FLUID_DENSITY'] = density
    assert zero_density_error < 1e-15
    dc = [dict(e,frequency=0.) for e in all_es]
    dc_error = float(np.max(abs(model.predict(dc)-baseline.predict(dc))))
    assert dc_error < 1e-15
    omega,force = 1e10,1e-9
    velocity = model.predict([dict(frequency=omega,force=force,component='sine')])[0]
    total_mass = globals_['PARTICLE_MASS']+half_displaced
    high_frequency_error = float(abs(omega*velocity*total_mass/force-1))
    assert high_frequency_error < 1e-3
    return dict(exterior_flow_traction_relative_error=force_error, basset_fourier_integral_relative_error=kernel_error,
                potential_flow_added_mass_relative_error=added_mass_error, minimum_real_fluid_impedance=minimum_real_impedance,
                zero_fluid_density_error=zero_density_error, dc_stokes_error=dc_error,
                high_frequency_inertial_limit_error=high_frequency_error)



if __name__ == '__main__':
    main()
