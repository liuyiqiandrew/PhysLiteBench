"""Validate entropy-anomaly revision 3; regenerate calibration only with --generate."""
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
    seed = 9332
    if generate:
        noise = np.random.default_rng(seed).normal(0, sigma, len(inputs))
        records = [dict(input=e, value=float(v), sigma=float(sigma)) for e, v in zip(inputs, noiseless+noise)]
        for folder in ['environment', 'tests']:
            path = task/folder/'data/calibration.json'
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(records, indent=2)+'\n')
    records = json.loads((task/'environment/data/calibration.json').read_text())
    assert records == json.loads((task/'tests/data/calibration.json').read_text())
    report = {'revision':3, 'seed':seed, 'noise_trials':noise_trials, 'controls':{}, 'noise':{}}
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
    parser.add_argument('--output', type=Path, default=ROOT/'jobs/entropy-anomaly-r3-validation/summary.json')
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
    mass_error = float(np.max(abs(ref.predict(es,model.friction,epsilon=.003)-kinetic)))
    basis_error = float(np.max(abs(ref.predict(es,model.friction,degree=20,spatial=61)-kinetic)))
    assert kinetic_error < 2e-6 and mass_error < 2e-6 and basis_error < 2e-7
    globals_ = model.predict.__func__.__globals__
    state_function = globals_['positional_state']
    errors = dict(normalization=0.,constant_current=0.,uniform_bath=0.,isotropic_field=0.,
                  zero_field_closure=0.,field_reflection=0.,heat_balance=0.,force_work=0.)
    min_density = 1.; min_covariance = 1.; corners = []
    for gamma in [.7,1.1,1.6]:
        model.friction = baseline.friction = gamma
        for ratio in [.25,1.,8.]:
            for magnetic in [-5.,0.,5.]:
                e = dict(force_x=-1.5,force_y=1.2,drag_ratio=ratio,magnetic=magnetic,
                         temperature=.8,contrast=0.,wavenumber=3)
                friction_matrix = np.array([[gamma,-magnetic],[magnetic,gamma*ratio]])
                forces = np.array([e['force_x'],e['force_y']])
                exact = forces@np.linalg.solve(friction_matrix,forces)/e['temperature']
                errors['uniform_bath'] = max(errors['uniform_bath'],abs(model.predict([e])[0]-exact))
                e = dict(e,force_x=.3,force_y=.2,contrast=.65,temperature=1.4)
                temp,grad,rho,jx,jy,dx = state_function(e,gamma)
                min_density = min(min_density,float(rho.min()))
                errors['normalization'] = max(errors['normalization'],abs(rho.sum()*dx-1))
                exact_jx = (gamma*ratio*e['force_x']+magnetic*e['force_y'])/(gamma**2*ratio+magnetic**2)/(2*np.pi)
                errors['constant_current'] = max(errors['constant_current'],abs(jx-exact_jx))
                positional = np.sum((gamma*jx*jx+gamma*ratio*jy*jy)/(temp*rho))*dx
                work = np.sum((e['force_x']*jx+e['force_y']*jy)/temp)*dx
                errors['force_work'] = max(errors['force_work'],abs(positional-work))
                reflected = dict(e,magnetic=-magnetic,force_y=-e['force_y'])
                errors['field_reflection'] = max(errors['field_reflection'],abs(model.predict([e])[0]-model.predict([reflected])[0]))
                if magnetic == 0:
                    errors['zero_field_closure'] = max(errors['zero_field_closure'],abs(model.predict([e])[0]-baseline.predict([e])[0]))
                if ratio == 1:
                    e0 = dict(e,force_x=0.,force_y=0.)
                    exact = 6*gamma/(9*gamma**2+magnetic**2)*e['temperature']*e['wavenumber']**2*(1-np.sqrt(1-e['contrast']**2))
                    errors['isotropic_field'] = max(errors['isotropic_field'],abs(model.predict([e0])[0]-exact))
        for ratio in [.25,8.]:
            e = dict(force_x=-.4,force_y=.3,drag_ratio=ratio,magnetic=5.,temperature=.8,contrast=.65,wavenumber=3)
            corners.append(float(abs(model.predict([e])[0]-ref.predict([e],gamma)[0])))
    e = dict(force_x=.3,force_y=-.2,drag_ratio=.4,magnetic=3.,temperature=1.,contrast=.65,wavenumber=3)
    mass=.0005; gamma=1.1
    theta,temp,c = ref.kinetic_state(e,gamma,mass,degree=20,spatial=61)
    gradient = -e['temperature']*e['contrast']*e['wavenumber']*np.sin(theta)
    jx = np.sqrt(e['temperature']/mass)*c[1,0]
    jy = np.sqrt(e['temperature']/mass)*c[0,1]
    third = (e['temperature']/mass)**1.5*(4*c[1,0]+np.sqrt(6)*c[3,0]+np.sqrt(2)*c[1,2])
    moment_heat = float(np.mean((e['force_x']*jx+e['force_y']*jy)/temp-mass*third*gradient/(2*temp**2))*2*np.pi)
    direct_heat = ref.finite_mass_entropy(e,gamma,mass,degree=20,spatial=61)
    errors['heat_balance'] = abs(moment_heat-direct_heat)
    covariance = np.empty((len(theta),2,2))
    covariance[:,0,0] = 1+np.sqrt(2)*c[2,0]/c[0,0]-(c[1,0]/c[0,0])**2
    covariance[:,1,1] = 1+np.sqrt(2)*c[0,2]/c[0,0]-(c[0,1]/c[0,0])**2
    covariance[:,0,1] = covariance[:,1,0] = c[1,1]/c[0,0]-c[1,0]*c[0,1]/c[0,0]**2
    min_covariance = float(np.linalg.eigvalsh(covariance).min())
    negative_tail = {}
    for degree in [14,24,32]:
        theta,temp,c = ref.kinetic_state(e,gamma,mass,degree=degree,spatial=41)
        w = np.linspace(-8,8,801)
        basis = np.array([eval_hermitenorm(n,w)*np.exp(-gammaln(n+1)/2) for n in range(degree+1)])
        probability = np.array([c[n,0] for n in range(degree+1)]).T@basis*np.exp(-w*w/2)/np.sqrt(2*np.pi)
        negative_tail[str(degree)] = float(np.maximum(-probability,0).sum()*(w[1]-w[0])*2*np.pi/len(theta))
    assert negative_tail['32'] < negative_tail['24'] < negative_tail['14']
    assert min_covariance > 0 and min_density > 0
    assert max(errors.values()) < 1e-7 and max(corners) < 2e-6
    return dict(kinetic_reference_error=kinetic_error,mass_refinement_error=mass_error,
                space_velocity_refinement_error=basis_error,identity_errors=errors,
                kinetic_negative_x_tail_mass=negative_tail,minimum_conditional_velocity_covariance=min_covariance,
                allowed_corner_reference_error=max(corners),minimum_position_density=min_density)


if __name__ == '__main__':
    main()
