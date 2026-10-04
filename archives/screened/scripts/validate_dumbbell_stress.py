"""Validate dumbbell-stress revision 1; regenerate calibration only with --generate."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['dumbbell-stress']


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
    seed = 9329
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
    parser.add_argument('--output', type=Path, default=ROOT/'jobs/dumbbell-stress-validation/summary.json')
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
    model = oracle_class(); model.relaxation_time = ref.TRUE_PARAMETER
    baseline = baseline_class(); baseline.relaxation_time = model.relaxation_time
    all_es = [e for es in ref.hidden_inputs().values() for e in es]
    stochastic_reference_error = float(np.max(abs(model.predict(all_es)-ref.predict(all_es,model.relaxation_time))))
    quadrature_error = float(np.max(abs(ref.predict(all_es,model.relaxation_time,32)-ref.predict(all_es,model.relaxation_time,64))))
    assert stochastic_reference_error < 1e-11 and quadrature_error < 1e-11
    oracle_propagate = model.predict.__func__.__globals__['propagate']
    shortcut_propagate = baseline.predict.__func__.__globals__['propagate']
    shear_error = 0.
    extension_error = 0.
    rotation_error = 0.
    objectivity_error = 0.
    minimum_covariance_eigenvalue = 1.
    source_minimum = float('inf')
    for tau in [.4,.8,1.2]:
        for time in [.0,.1,1.,6.]:
            shear = .65
            physical = oracle_propagate(np.eye(2),[[0.,shear],[0.,0.]],time,tau)
            decay = np.exp(-time/tau)
            exact = np.array([[1+2*(shear*tau)**2*(1-decay*(1+time/tau)),shear*tau*(1-decay)],
                              [shear*tau*(1-decay),1.]])
            shear_error = max(shear_error,float(np.max(abs(physical-exact))))
            extension = .35
            physical = oracle_propagate(np.eye(2),np.diag([extension,-extension]),time,tau)
            rates = np.array([2*extension-1/tau,-2*extension-1/tau])
            exact = np.diag(np.exp(rates*time)+np.expm1(rates*time)/(tau*rates))
            extension_error = max(extension_error,float(np.max(abs(physical-exact))))
            initial = np.array([[1.5,.2],[.2,.5]])
            omega = 1.2
            angle = omega*time
            rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            exact = np.eye(2)+decay*rotation@(initial-np.eye(2))@rotation.T
            gradient = np.array([[0.,-omega],[omega,0.]])
            for propagate in [oracle_propagate,shortcut_propagate]:
                rotation_error = max(rotation_error,float(np.max(abs(propagate(initial,gradient,time,tau)-exact))))
        # Frame rotations must act on the connector covariance as a tensor in
        # both constitutive models; nonobjectivity is not the intended defect.
        angle = .63
        rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        initial = np.array([[.7,.3],[.3,1.8]])
        gradient = np.array([[.15,.8],[-.3,-.15]])
        for propagate in [oracle_propagate,shortcut_propagate]:
            c = propagate(initial,gradient,3.,tau)
            rotated = propagate(rotation@initial@rotation.T,rotation@gradient@rotation.T,3.,tau)
            objectivity_error = max(objectivity_error,float(np.max(abs(rotated-rotation@c@rotation.T))))
        # Allowed maximum strain still leaves the shortcut's covariance source
        # positive, so its constitutive failure need not produce invalid C.
        for sign in [-1.,1.]:
            gradient = np.array([[.35,-1.2*sign],[1.2*sign,-.35]])
            source_minimum = min(source_minimum,float(np.linalg.eigvalsh(np.eye(2)/tau+gradient+gradient.T)[0]))
            for propagate in [oracle_propagate,shortcut_propagate]:
                c = propagate(initial,gradient,6.,tau)
                minimum_covariance_eigenvalue = min(minimum_covariance_eigenvalue,float(np.linalg.eigvalsh(c)[0]))
    assert max(shear_error,extension_error,rotation_error,objectivity_error) < 1e-11
    assert minimum_covariance_eigenvalue > 0 and source_minimum > 0
    equilibrium_error = float(np.max(abs(oracle_propagate(np.eye(2),np.zeros((2,2)),6.,1.2)-np.eye(2))))
    assert equilibrium_error < 1e-12
    # Carrying the state through a split interval cannot change a constant flow.
    segmentation_error = 0.
    gradient = np.array([[.2,.7],[-.3,-.2]])
    for propagate in [oracle_propagate,shortcut_propagate]:
        whole = propagate(np.eye(2),gradient,4.,.8)
        split = propagate(propagate(np.eye(2),gradient,1.3,.8),gradient,2.7,.8)
        segmentation_error = max(segmentation_error,float(np.max(abs(whole-split))))
    assert segmentation_error < 1e-11
    return dict(gaussian_noise_transport_error=stochastic_reference_error,
                quadrature32_vs64_error=quadrature_error, exact_simple_shear_error=shear_error,
                exact_planar_extension_error=extension_error, rigid_rotation_relaxation_error=rotation_error,
                tensor_rotation_covariance_error=objectivity_error,
                minimum_covariance_eigenvalue=minimum_covariance_eigenvalue,
                minimum_shortcut_covariance_source=source_minimum,
                quiescent_equilibrium_error=equilibrium_error, segmentation_error=segmentation_error)


if __name__ == '__main__':
    main()
