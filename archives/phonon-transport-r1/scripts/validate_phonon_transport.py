"""Validate phonon-transport revision 1; regenerate calibration only with --generate."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['phonon-transport']


def module(path):
    spec = importlib.util.spec_from_file_location('candidate_'+str(abs(hash(str(path)))), path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def scale_for(name, experiments, truth):
    return max(float(np.sqrt(np.mean(truth**2))),1e-6)


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
    inputs = ref.calibration_inputs()
    noiseless = ref.predict(inputs, ref.TRUE_PARAMETER)
    sigma = .006*max(np.max(np.abs(noiseless)),1e-10)
    seed = 13119
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
        assert calibration_delta < 5e-6
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
    parser.add_argument('--output', type=Path, default=ROOT/'jobs/phonon-transport-validation/summary.json')
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


def independent_checks(ref,oracle_class,baseline_class):
    oracle=module(ROOT/'tasks/phonon-transport/solution/model.py')
    baseline=module(ROOT/'scripts/phonon_transport_baseline.py')
    es=[e for group in ref.hidden_inputs().values() for e in group]
    r=ref.TRUE_PARAMETER
    truth=ref.predict(es,r)
    error=float(np.max(abs(oracle.predict_at(es,r)-truth)))
    quadrature=float(np.max(abs(ref.predict(es,r,order=120)-truth)))
    truncation=float(np.max(abs(oracle.predict_at(es,r,order=96)-oracle.predict_at(es,r))))
    mu,w,p0,p1=ref.angles(80)
    normal=p0+p1-np.eye(80);resistive=p0-np.eye(80)
    energy_residual=max(float(np.max(abs(w@operator))) for operator in [normal,resistive])
    momentum_residual=float(np.max(abs((w*mu)@normal)))
    dissipative=np.diag(np.sqrt(w))@normal@np.diag(1/np.sqrt(w))
    maximum_collision_eigenvalue=float(max(np.linalg.eigvalsh((dissipative+dissipative.T)/2)))
    homogeneous=[ref.reading(t,0.,n,(.05,.05,.05),l) for t in [0.,.2,2.,8.] for n in [0.,3.] for l in [0,1,2]]
    expected=np.array([e['initial'][e['moment']]/(2*e['moment']+1)*np.exp(-e['time']*(0 if e['moment']==0 else r if e['moment']==1 else r+e['normal_rate'])) for e in homogeneous])
    homogeneous_error=float(np.max(abs(oracle.predict_at(homogeneous,r)-expected)))
    extreme=[ref.reading(t,k,n,(.05,.05,.05),l,q) for t in [0.,8.] for k in [0.,3.] for n in [0.,3.] for l in [0,1,2] for q in ['cosine','sine']]
    corner_error=0.;zero_normal_error=0.;ballistic_error=0.
    for rate in [.15,.6]:
        corner_error=max(corner_error,float(np.max(abs(oracle.predict_at(extreme,rate)-ref.predict(extreme,rate,order=120)))))
        zero=[dict(e,normal_rate=0.) for e in extreme]
        zero_normal_error=max(zero_normal_error,float(np.max(abs(oracle.predict_at(zero,rate)-baseline.predict_at(zero,rate)))))
    # At zero collisions the reference is direct free angular streaming.
    ballistic=[ref.reading(t,1.7,0.,(.12,0.,0.),0) for t in [.0,.3,1.,4.]]
    expected=np.array([.12*np.sinc(1.7*e['time']/np.pi) for e in ballistic])
    ballistic_error=float(np.max(abs(oracle.predict_at(ballistic,0.)-expected)))
    assert error<1e-10 and quadrature<1e-10 and truncation<1e-10 and corner_error<1e-10
    assert energy_residual<1e-12 and momentum_residual<1e-12 and maximum_collision_eigenvalue<1e-12
    assert homogeneous_error<1e-12 and zero_normal_error<1e-12 and ballistic_error<1e-12
    return dict(angular_reference_error=error,angular_80_vs_120_error=quadrature,
                legendre_64_vs_96_error=truncation,parameter_and_input_corner_error=corner_error,
                collision_energy_residual=energy_residual,normal_collision_momentum_residual=momentum_residual,
                maximum_weighted_normal_collision_eigenvalue=maximum_collision_eigenvalue,
                homogeneous_mode_error=homogeneous_error,zero_normal_control_agreement=zero_normal_error,
                collisionless_sinc_error=ballistic_error)


if __name__ == '__main__':
    main()
