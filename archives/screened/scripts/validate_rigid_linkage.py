"""Validate rigid-linkage revision 1; regenerate calibration only with --generate."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['rigid-linkage']


def module(path):
    spec = importlib.util.spec_from_file_location('candidate_'+str(abs(hash(str(path)))), path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def scale_for(name, experiments, truth):
    return np.ones_like(truth)


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
    seed = 9324
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
    parser.add_argument('--output', type=Path, default=ROOT/'jobs/rigid-linkage-validation/summary.json')
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
    from scipy.special import i0e,i1e
    model = oracle_class(); model.stiffness = ref.TRUE_PARAMETER
    all_es = [e for es in ref.hidden_inputs().values() for e in es]
    expected = ref.predict(all_es,model.stiffness)
    reference_error = float(np.max(abs(model.predict(all_es)-expected)))
    assert reference_error < 1e-8
    lower_order = ref.predict(all_es,model.stiffness,order=128)
    angular_error = float(np.max(abs(lower_order-expected)))
    assert angular_error < 1e-8
    momentum_error = 0.
    for ratio,relative in [(.02,0.),(.02,np.pi/2),(.2,.7),(1.,1.3)]:
        inverse = np.linalg.inv(ref.inertia(.3,.3+relative,ratio))
        a,b,c = inverse[0,0],inverse[0,1],inverse[1,1]
        def outer(p):
            return quad(lambda q:np.exp(-.5*(a*p*p+2*b*p*q+c*q*q)),
                        -12.,12.,points=[-b*p/c],epsabs=3e-10,epsrel=3e-10)[0]
        integral = quad(outer,-12.,12.,epsabs=3e-9,epsrel=3e-9)[0]
        exact = 2*np.pi*np.sqrt(ratio+np.sin(relative)**2)
        momentum_error = max(momentum_error,float(abs(integral/exact-1)))
    assert momentum_error < 1e-7
    # No-field global rotation is uniform, but the relative angle is not.
    isotropic = [ref.experiment(False,factors=(0.,0.),mass_ratio=r,harmonic=h)
                 for r in [.02,.2,1.] for h in [(1,0),(0,1),(2,0),(0,2)]]
    rotation_error = float(np.max(abs(model.predict(isotropic))))
    assert rotation_error < 1e-12
    weak_mass = ref.experiment(False,factors=(0.,0.),mass_ratio=1e-10)
    small_mass_error = float(abs(model.predict([weak_mass])[0]+1/3))
    assert small_mass_error < 1e-4
    strong_mass = ref.experiment(False,factors=(0.,0.),mass_ratio=1e8)
    large_mass_error = float(abs(model.predict([strong_mass])[0]))
    assert large_mass_error < 1e-8
    # Clamping removes the first momentum before equilibration.
    clamp = [ref.experiment(True,factors=(1.,f),mass_ratio=r,harmonic=(0,1))
             for r in [.02,1.] for f in [.2,1.,2.]]
    z = model.stiffness*np.array([e['factors'][1] for e in clamp])
    clamped_error = float(np.max(abs(model.predict(clamp)-i1e(z)/i0e(z))))
    assert clamped_error < 1e-12
    one = [dict(e,harmonic=[0,0],phase=0.) for e in all_es]
    normalization_error = float(np.max(abs(model.predict(one)-1)))
    assert normalization_error < 1e-14
    return dict(cartesian_momentum_reference_error=reference_error, angular_128_vs_192_error=angular_error,
                direct_momentum_quadrature_relative_error=momentum_error, rotational_invariance_error=rotation_error,
                small_mass_limit_error=small_mass_error, large_mass_limit_error=large_mass_error,
                clamped_von_mises_error=clamped_error, normalization_error=normalization_error)



if __name__ == '__main__':
    main()
