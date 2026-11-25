"""Validate colored-spin-memory revision 2; regenerate calibration only with --generate."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['colored-spin-memory']


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
    seed = 9323
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
            assert metrics['hidden']['changed_axis'] > meta['prediction_limit'], metrics
            assert max(v for k,v in metrics['hidden'].items() if k != 'changed_axis') < meta['prediction_limit'], metrics
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
        scores = {key: [] for key in hidden}
        for p in [min(fitted), max(fitted)]:
            model = cls()
            setattr(model, ref.PARAMETER, p)
            for key, es in hidden.items():
                score = float(np.sqrt(np.mean(((model.predict(es)-truths[key])/scale_for(name, es, truths[key]))**2)))
                scores[key].append(score)
        sensitivity[label] = {key: {'min':min(values), 'max':max(values)} for key,values in scores.items()}
    assert max(v['max'] for v in sensitivity['oracle'].values()) < meta['prediction_limit']
    assert sensitivity['shortcut']['changed_axis']['min'] > meta['prediction_limit']
    assert max(v['max'] for k,v in sensitivity['shortcut'].items() if k != 'changed_axis') < meta['prediction_limit']
    report['hidden_parameter_extrema_check'] = sensitivity
    report['expected_shortcut_failures'] = ['changed_axis']

    report['independent_checks'] = independent_checks(ref, oracle_class, baseline_class)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tasks', nargs='*', choices=NAMES)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--noise-trials', type=int, default=256)
    parser.add_argument('--output', type=Path, default=ROOT/'jobs/colored-spin-memory-r2-validation/summary.json')
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
    from numpy.polynomial.hermite import hermgauss
    from scipy.linalg import expm
    from scipy.spatial.transform import Rotation
    model = oracle_class(); model.noise_width = ref.TRUE_PARAMETER
    baseline = baseline_class(); baseline.noise_width = model.noise_width
    all_es = [e for es in ref.hidden_inputs().values() for e in es]
    physical = model.predict(all_es)
    coarse = ref.predict(all_es, model.noise_width, cells=321)
    fine = ref.predict(all_es, model.noise_width, cells=641)
    coarse_error = float(np.max(abs(coarse-physical)))
    fine_error = float(np.max(abs(fine-physical)))
    assert fine_error < 3e-5 and fine_error < .3*coarse_error
    globals_ = model.predict.__func__.__globals__
    old_modes = globals_['MODES']
    globals_['MODES'] = 60
    spectral_error = float(np.max(abs(model.predict(all_es)-physical)))
    globals_['MODES'] = old_modes
    assert spectral_error < 1e-10
    # Large allowed controls and the largest fitted-parameter bound still converge.
    corners = [dict(preparation=[.7,2.2],readout=[-1.3,-2.1],
                    segments=[dict(duration=2.,field=[4.,-4.,4.]),
                              dict(duration=2.,field=[-4.,4.,-4.])]),
               dict(preparation=[1.,1.5],readout=[-1.,1.2],
                    segments=[dict(duration=4.,field=[0.,4.,0.])])]
    model.noise_width = 1.6
    corner_coarse = model.predict(corners)
    globals_['MODES'] = 60
    corner_error = float(np.max(abs(model.predict(corners)-corner_coarse)))
    globals_['MODES'] = old_modes
    model.noise_width = ref.TRUE_PARAMETER
    assert corner_error < 1e-10
    # The Gaussian phase has variance equal to the double covariance integral.
    from scipy.integrate import quad
    phase_variance_error = 0.
    for t in [.05,.5,2.,4.]:
        variance = 2*model.noise_width**2*quad(lambda u:(t-u)*np.exp(-u),0.,t,epsabs=1e-13)[0]
        analytical = 2*model.noise_width**2*(t+np.expm1(-t))
        phase_variance_error = max(phase_variance_error, abs(variance-analytical))
    assert phase_variance_error < 1e-12
    ramsey = ref.calibration_inputs()[::10]
    tiny_detuning = [dict(e,segments=[dict(e['segments'][0],field=[0.,0.,1e-13])]) for e in ramsey]
    # Tiny deterministic z detuning forces the general Hermite path without changing the probability appreciably.
    hermite_ramsey_error = float(np.max(abs(model.predict(tiny_detuning)-ref.predict(ramsey,model.noise_width))))
    assert hermite_ramsey_error < 1e-11
    model.noise_width = baseline.noise_width = 0.
    deterministic_error = float(np.max(abs(model.predict(all_es)-baseline.predict(all_es))))
    assert deterministic_error < 1e-8
    # Each reduced channel is a complete map, not a propagation of one selected
    # initial spin. It must agree with the physical solution for all basis states.
    model.noise_width = baseline.noise_width = ref.TRUE_PARAMETER
    channel_error = 0.
    channel_largest_singular_value = 0.
    preparations = [[np.pi/2,np.pi/2],[0.,-np.pi/2],[0.,0.]]
    readouts = [[np.pi/2,-np.pi/2],[0.,np.pi/2],[0.,0.]]
    for field,duration in [([2.5,0.,.3],1.),([0.,2.8,-.4],3.),([0.,0.,0.],2.)]:
        channel = baseline._segment_channel(field,duration)
        channel_largest_singular_value = max(channel_largest_singular_value, float(np.linalg.svd(channel,compute_uv=False)[0]))
        experiments = [dict(preparation=p,readout=r,segments=[dict(duration=duration,field=field)])
                       for p in preparations for r in readouts]
        predicted = (channel.T.reshape(-1)+1)/2
        channel_error = max(channel_error,float(np.max(abs(model.predict(experiments)-predicted))))
    assert channel_error < 1e-12 and channel_largest_singular_value <= 1+1e-12
    # Splitting one constant control into two must not change the physical result.
    whole = dict(preparation=[0.,0.],readout=[0.,0.],segments=[dict(duration=3.,field=[2.5,0.,.3])])
    split = dict(whole,segments=[dict(duration=1.,field=[2.5,0.,.3]),dict(duration=2.,field=[2.5,0.,.3])])
    oracle_segmentation_error = float(abs(np.diff(model.predict([whole,split]))[0]))
    closure_segmentation_difference = float(abs(np.diff(baseline.predict([whole,split]))[0]))
    assert oracle_segmentation_error < 1e-12 and closure_segmentation_difference > .005
    # For tau tending to infinity, the OU field is constant during the shot.
    model.noise_width = ref.TRUE_PARAMETER
    original_tau = globals_['CORRELATION_TIME']
    globals_['CORRELATION_TIME'] = 1e8
    subset = [es[-1] for es in ref.hidden_inputs().values()]
    slow = model.predict(subset)
    nodes, weights = hermgauss(80)
    exact = []
    def pulse(v, spec):
        azimuth,angle = spec
        return Rotation.from_rotvec(angle*np.array([np.cos(azimuth),np.sin(azimuth),0.])).apply(v)
    for e in subset:
        value = 0.
        for z,w in zip(nodes,weights/np.sqrt(np.pi)):
            v = pulse(np.array([0.,0.,1.]),e['preparation'])
            for segment in e['segments']:
                field = np.array(segment['field'],dtype=float)
                field[2] += np.sqrt(2)*model.noise_width*z
                v = Rotation.from_rotvec(field*segment['duration']).apply(v)
            value += w*(1+pulse(v,e['readout'])[2])/2
        exact.append(value)
    quasistatic_error = float(np.max(abs(slow-exact)))
    globals_['CORRELATION_TIME'] = original_tau
    assert quasistatic_error < 1e-7
    # In the fast-noise limit sigma^2*tau=D, the Bloch generator has damping D.
    diffusion = .4
    e = dict(preparation=[0.,0.],readout=[0.,0.],segments=[dict(duration=.2,field=[3.,0.,.2])])
    field = np.array(e['segments'][0]['field'])
    matrix = globals_['cross_matrix'](field)-np.diag([diffusion,diffusion,0.])
    markov = (1+(expm(matrix*.2)@np.array([0.,0.,1.]))[2])/2
    globals_['CORRELATION_TIME'] = .0002
    model.noise_width = np.sqrt(diffusion/.0002)
    white_error = float(abs(model.predict([e])[0]-markov))
    globals_['CORRELATION_TIME'] = original_tau
    assert white_error < 5e-5
    return dict(finite_volume_321_error=coarse_error, finite_volume_641_error=fine_error,
                hermite_40_vs_60_error=spectral_error, allowed_corner_spectral_error=corner_error,
                covariance_integral_error=phase_variance_error, hermite_ramsey_error=hermite_ramsey_error,
                zero_noise_error=deterministic_error, quasistatic_limit_error=quasistatic_error,
                white_noise_limit_error=white_error, segment_channel_basis_error=channel_error,
                channel_largest_singular_value=channel_largest_singular_value,
                oracle_segmentation_error=oracle_segmentation_error,
                closure_segmentation_difference=closure_segmentation_difference)



if __name__ == '__main__':
    main()
