"""Validate entropy-anomaly revision 1; regenerate calibration only with --generate."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['entropy-anomaly']


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
    sigma = .01*max(np.max(np.abs(noiseless)), 1e-10)
    seed = 9330
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
    parser.add_argument('--output', type=Path, default=ROOT/'jobs/entropy-anomaly-validation/summary.json')
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
    from scipy.special import eval_hermitenorm, gammaln
    model = oracle_class(); model.friction = ref.TRUE_PARAMETER
    baseline = baseline_class(); baseline.friction = model.friction
    es = [e for group in ref.hidden_inputs().values() for e in group]
    expected = model.predict(es)
    kinetic = ref.predict(es,model.friction)
    kinetic_error = float(np.max(abs(kinetic-expected)))
    refined_mass = ref.predict(es,model.friction,epsilon=.002)
    mass_error = float(np.max(abs(refined_mass-kinetic)))
    refined_basis = ref.predict(es,model.friction,points=127,modes=64)
    basis_error = float(np.max(abs(refined_basis-kinetic)))
    assert kinetic_error < 2e-6 and mass_error < 2e-6 and basis_error < 2e-7
    state_function = model.predict.__func__.__globals__['positional_state']
    conservation_error = 0.
    constant_force_current_error = 0.
    uniform_error = 0.
    no_force_error = 0.
    no_force_entropy_error = 0.
    reversal_error = 0.
    cancellation_error = 0.
    minimum_density = 1.
    negatives = {}
    corner_errors = []
    for gamma in [.7,1.1,1.6]:
        model.friction = gamma
        for temperature in [.8,1.4]:
            for force in [-1.5,0.,1.5]:
                e = dict(force=force,temperature=temperature,contrast=0.,wavenumber=3)
                uniform_error = max(uniform_error,abs(model.predict([e])[0]-force**2/(gamma*temperature)))
                # Finite-mass heat itself also has this exact uniform-bath limit.
                uniform_error = max(uniform_error,abs(ref.finite_mass_entropy(e,gamma,.01)-force**2/(gamma*temperature)))
        e = dict(force=0.,temperature=1.,contrast=.65,wavenumber=3)
        temp,grad,rho,current,dx = state_function(e,gamma)
        exact_density = (1/temp)/(np.sum(1/temp)*dx)
        no_force_error = max(no_force_error,float(np.max(abs(rho-exact_density))))
        exact_entropy = e['temperature']*e['wavenumber']**2*(1-np.sqrt(1-e['contrast']**2))/(2*gamma)
        no_force_entropy_error = max(no_force_entropy_error,float(abs(model.predict([e])[0]-exact_entropy)))
        e = dict(force=1.5,temperature=.8,contrast=.65,wavenumber=3)
        temp,grad,rho,current,dx = state_function(e,gamma)
        minimum_density = min(minimum_density,float(rho.min()))
        conservation_error = max(conservation_error,abs(np.sum(rho)*dx-1))
        constant_force_current_error = max(constant_force_current_error,abs(current-e['force']/(2*np.pi*gamma)))
        reflected = dict(e,force=-e['force'])
        reversal_error = max(reversal_error,float(abs(model.predict([e])[0]-model.predict([reflected])[0])))
        corner_errors.append(float(abs(ref.predict([e],gamma)[0]-model.predict([e])[0])))
    e = dict(force=.5,temperature=1.,contrast=.65,wavenumber=3)
    # The reference is a velocity spectral approximation. Its small negative
    # tail mass must converge away, not be confused with physical negativity.
    for modes in [24,48,64]:
        mass=.0001; gamma=1.1
        x,temp,coeff = ref.kinetic_state(e,gamma,mass,points=127,modes=modes)
        w = np.linspace(-8,8,1001)
        basis = np.array([eval_hermitenorm(n,w)*np.exp(-gammaln(n+1)/2) for n in range(modes)])
        probability = (coeff.T@basis)*np.exp(-w*w/2)/np.sqrt(2*np.pi)
        negative_mass = float(np.maximum(-probability,0).sum()*(w[1]-w[0])*2*np.pi/len(x))
        negatives[str(modes)] = negative_mass
        gradient=-e['temperature']*e['contrast']*e['wavenumber']*np.sin(e['wavenumber']*x)
        current=coeff[1]*np.sqrt(e['temperature']/mass)
        third=(3*coeff[1]+np.sqrt(6)*coeff[3])*(e['temperature']/mass)**1.5
        moment_heat=float(np.mean(e['force']*current/temp-mass*third*gradient/(2*temp**2))*2*np.pi)
        direct_heat=ref.finite_mass_entropy(e,gamma,mass,points=127,modes=modes)
        cancellation_error=max(cancellation_error,abs(moment_heat-direct_heat))
    assert negatives['64'] < 1e-9 and negatives['64'] < negatives['48'] < negatives['24']
    assert max(uniform_error,no_force_error,no_force_entropy_error,conservation_error,constant_force_current_error,reversal_error) < 1e-10
    assert cancellation_error < 1e-8 and max(corner_errors) < 2e-6
    assert minimum_density > 0
    # The limiting rate scales exactly as 1/gamma; this does not hold at fixed
    # finite mass and provides an additional check on the limiting prediction.
    model.friction=.7; a=model.predict(es)*.7
    model.friction=1.6; b=model.predict(es)*1.6
    friction_scaling_error=float(np.max(abs(a-b)))
    assert friction_scaling_error < 1e-12
    return dict(kinetic_reference_error=kinetic_error,mass_refinement_error=mass_error,
                space_velocity_refinement_error=basis_error,uniform_bath_error=uniform_error,
                zero_force_density_error=no_force_error,zero_force_entropy_error=no_force_entropy_error,normalization_error=conservation_error,
                exact_current_error=constant_force_current_error,force_reversal_error=reversal_error,
                direct_heat_vs_third_moment_error=cancellation_error,kinetic_negative_tail_mass=negatives,
                allowed_corner_reference_error=max(corner_errors),minimum_position_density=minimum_density,
                friction_scaling_error=friction_scaling_error)


if __name__ == '__main__':
    main()
