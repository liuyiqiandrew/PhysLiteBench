"""Validate entropy-anomaly revision 2; regenerate calibration only with --generate."""
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
    sigma = .008*max(np.max(np.abs(noiseless)), 1e-10)
    seed = 9331
    if generate:
        noise = np.random.default_rng(seed).normal(0, sigma, len(inputs))
        records = [dict(input=e, value=float(v), sigma=float(sigma)) for e, v in zip(inputs, noiseless+noise)]
        for folder in ['environment', 'tests']:
            path = task/folder/'data/calibration.json'
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(records, indent=2)+'\n')
    records = json.loads((task/'environment/data/calibration.json').read_text())
    assert records == json.loads((task/'tests/data/calibration.json').read_text())
    report = {'revision':2, 'seed':seed, 'noise_trials':noise_trials, 'controls':{}, 'noise':{}}
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
    parser.add_argument('--output', type=Path, default=ROOT/'jobs/entropy-anomaly-r2-validation/summary.json')
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
    refined_basis = ref.predict(es,model.friction,points=95,modes=48)
    basis_error = float(np.max(abs(refined_basis-kinetic)))
    assert kinetic_error < 2e-6 and mass_error < 2e-6 and basis_error < 2e-7
    global_model = model.predict.__func__.__globals__
    state_function = global_model['positional_state']
    normalization_error = current_error = uniform_error = zero_force_error = 0.
    reversal_error = heat_balance_error = isotropic_error = 0.
    minimum_density = 1.
    corner_errors = []
    for gamma in [.7,1.1,1.6]:
        model.friction = baseline.friction = gamma
        for ratio in [.25,1.,8.]:
            e = dict(force_x=-1.5,force_y=1.2,drag_ratio=ratio,temperature=.8,contrast=0.,wavenumber=3)
            exact = (e['force_x']**2+e['force_y']**2/ratio)/(gamma*e['temperature'])
            uniform_error=max(uniform_error,float(abs(model.predict([e])[0]-exact)),
                              abs(ref.finite_mass_entropy(e,gamma,.002)-exact))
            e = dict(e,force_x=0.,force_y=0.,contrast=.65,temperature=1.4)
            coefficient=.5*(1/gamma+1/(gamma+2*gamma*ratio))
            exact=coefficient*e['temperature']*e['wavenumber']**2*(1-np.sqrt(1-e['contrast']**2))
            zero_force_error=max(zero_force_error,float(abs(model.predict([e])[0]-exact)))
            temp,gradient,rho,current,dx=state_function(e,gamma)
            minimum_density=min(minimum_density,float(rho.min()))
            normalization_error=max(normalization_error,abs(np.sum(rho)*dx-1))
        e=dict(force_x=.6,force_y=.4,drag_ratio=1.,temperature=1.2,contrast=.6,wavenumber=2)
        isotropic_error=max(isotropic_error,float(abs(model.predict([e])[0]-baseline.predict([e])[0])))
        for ratio in [.25,8.]:
            e=dict(e,drag_ratio=ratio,temperature=.8,contrast=.65,wavenumber=3)
            corner_errors.append(float(abs(model.predict([e])[0]-ref.predict([e],gamma)[0])))
            temp,grad,rho,current,dx=state_function(e,gamma)
            current_error=max(current_error,abs(current-e['force_x']/(2*np.pi*gamma)))
            reversal_error=max(reversal_error,float(abs(model.predict([e])[0]-model.predict([dict(e,force_x=-e['force_x'],force_y=-e['force_y'])])[0])))
    e=dict(force_x=.5,force_y=.4,drag_ratio=4.,temperature=1.,contrast=.65,wavenumber=3)
    mass=.0005; gamma=1.1
    x,temp,(c0,c1,c2)=ref.kinetic_state(e,gamma,mass,points=95,modes=48)
    gradient=-e['temperature']*e['contrast']*e['wavenumber']*np.sin(e['wavenumber']*x)
    jx=np.sqrt(e['temperature']/mass)*c0[1]
    jy=np.sqrt(e['temperature']/mass)*c1[0]
    third=(e['temperature']/mass)**1.5*(4*c0[1]+np.sqrt(6)*c0[3]+np.sqrt(2)*c2[1])
    moment_heat=float(np.mean((e['force_x']*jx+e['force_y']*jy)/temp-mass*third*gradient/(2*temp**2))*2*np.pi)
    direct_heat=ref.finite_mass_entropy(e,gamma,mass,points=95,modes=48)
    heat_balance_error=abs(moment_heat-direct_heat)
    # Conditional y variance is a physical positive second moment, while the
    # x-velocity marginal's tiny spectral negative tail must converge away.
    conditional_y_variance=e['temperature']/mass*(c0[0]+np.sqrt(2)*c2[0])/c0[0]-(e['force_y']/(gamma*e['drag_ratio']))**2
    negatives={}
    for modes in [24,48,64]:
        x,temp,(c0,c1,c2)=ref.kinetic_state(e,gamma,mass,points=95,modes=modes)
        w=np.linspace(-8,8,1001)
        basis=np.array([eval_hermitenorm(n,w)*np.exp(-gammaln(n+1)/2) for n in range(modes)])
        probability=(c0.T@basis)*np.exp(-w*w/2)/np.sqrt(2*np.pi)
        negatives[str(modes)]=float(np.maximum(-probability,0).sum()*(w[1]-w[0])*2*np.pi/len(x))
    assert negatives['64'] < 1e-9 and negatives['64'] < negatives['48'] < negatives['24']
    assert float(conditional_y_variance.min()) > 0 and minimum_density > 0
    assert max(uniform_error,zero_force_error,normalization_error,current_error,reversal_error,isotropic_error) < 1e-9
    assert heat_balance_error < 1e-7 and max(corner_errors) < 2e-6
    model.friction=.7;a=model.predict(es)*.7
    model.friction=1.6;b=model.predict(es)*1.6
    friction_scaling_error=float(np.max(abs(a-b)))
    assert friction_scaling_error < 1e-12
    return dict(kinetic_reference_error=kinetic_error,mass_refinement_error=mass_error,
                space_velocity_refinement_error=basis_error,uniform_bath_error=uniform_error,
                exact_zero_force_entropy_error=zero_force_error,normalization_error=normalization_error,
                exact_current_error=current_error,force_reversal_error=reversal_error,
                equal_drag_closure_error=isotropic_error,direct_heat_vs_third_moment_error=heat_balance_error,
                kinetic_negative_x_tail_mass=negatives,minimum_conditional_y_variance=float(conditional_y_variance.min()),
                allowed_corner_reference_error=max(corner_errors),minimum_position_density=minimum_density,
                friction_scaling_error=friction_scaling_error)


if __name__ == '__main__':
    main()
