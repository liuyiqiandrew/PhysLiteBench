"""Validate exclusion-current revision 1; regenerate calibration only with --generate."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['exclusion-current']


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
    seed = 9326
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
    parser.add_argument('--output', type=Path, default=ROOT/'jobs/exclusion-current-validation/summary.json')
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
    from scipy.linalg import expm
    from scipy.special import expit
    model = oracle_class(); model.rate = ref.TRUE_PARAMETER
    baseline = baseline_class(); baseline.rate = model.rate
    all_es = [e for es in ref.hidden_inputs().values() for e in es]
    uniformization_error = float(np.max(abs(model.predict(all_es)-ref.predict(all_es,model.rate))))
    assert uniformization_error < 1e-10
    # Occupation means close as a two-dimensional linear system. This gives
    # boundary-count differences without constructing any counting generator.
    boundary_error = 0.
    short_time_error = 0.
    marginal_error = 0.
    symmetry_error = 0.
    stationary_error = 0.
    for bias in [-8.,-3.,0.,3.,8.]:
        filling = expit(bias/2)
        mean = np.array([(1+filling)/3,(2-filling)/3])
        covariance = np.diag(mean*(1-mean))
        covariance[0,1] = covariance[1,0] = -(2*filling-1)**2/18
        for rate in [.5,1.3,2.]:
            model.rate = baseline.rate = rate
            events,probability = ref.jump_process(bias,rate)
            generator = np.zeros((4,4))
            for j,i,k,dq in events:
                generator[i,j] += k
                generator[j,j] -= k
            stationary_error = max(stationary_error,float(np.max(abs(generator@probability))))
            for duration in [.0,.01,1.,12.]:
                transition = expm(rate*np.array([[-2.,1.],[1.,-2.]])*duration)
                for coefficients,weights in [([1.,0.],[1.,-1.,0.]),([0.,1.],[0.,1.,-1.]),([1.,1.],[1.,0.,-1.])]:
                    c = np.array(coefficients)
                    exact = 2*c@(covariance-transition@covariance)@c
                    e = dict(bias=bias,duration=duration,weights=weights,statistic='variance')
                    boundary_error = max(boundary_error,float(abs(model.predict([e])[0]-exact)))
                es = [dict(bias=bias,duration=duration,weights=list(w),statistic='variance') for w in np.eye(3)]
                marginal_error = max(marginal_error,float(np.max(abs(model.predict(es)-baseline.predict(es)))))
            e = dict(bias=bias,duration=1e-6,weights=[.7,-.3,.8],statistic='variance')
            shot_rate = sum(probability[j]*k*np.dot(e['weights'],dq)**2 for j,i,k,dq in events)
            short_time_error = max(short_time_error,float(abs(model.predict([e])[0]/e['duration']/shot_rate-1)))
            e['duration'] = 3.
            symmetry_error = max(symmetry_error,float(abs(model.predict([e])[0]-model.predict([dict(e,bias=-bias)])[0])))
    assert boundary_error < 1e-10 and marginal_error < 1e-10
    assert short_time_error < 1e-5 and symmetry_error < 1e-10 and stationary_error < 1e-12
    model.rate = ref.TRUE_PARAMETER
    # The three currents differ only by bounded stored particle numbers. Their
    # long-time noise rates agree; adding all three gives nine times one rate.
    duration = 1000.
    e = dict(bias=4.,duration=duration,weights=[1.,0.,0.],statistic='variance')
    single = model.predict([e])[0]
    summed = model.predict([dict(e,weights=[1.,1.,1.])])[0]
    conservation_ratio_error = float(abs(summed/(9*single)-1))
    assert conservation_ratio_error < .003
    # At equilibrium the long-time particle-current noise obeys 2*dJ/db.
    equilibrium_noise = model.predict([dict(e,bias=0.)])[0]/duration
    equilibrium_response_error = float(abs(equilibrium_noise/(model.rate/6)-1))
    assert equilibrium_response_error < .003
    mean_e = dict(bias=5.,duration=3.,weights=[.4,-.7,1.],statistic='mean')
    uniform_mean = ref.moments(mean_e,model.rate)[0]
    current_error = float(abs(uniform_mean-model.predict([mean_e])[0]))
    assert current_error < 1e-11
    return dict(uniformization_max_error=uniformization_error, occupancy_continuity_error=boundary_error,
                single_counter_agreement=marginal_error, short_time_shot_rate_relative_error=short_time_error,
                particle_hole_symmetry_error=symmetry_error, stationary_probability_error=stationary_error,
                long_time_current_conservation_relative_error=conservation_ratio_error,
                equilibrium_noise_response_relative_error=equilibrium_response_error,
                stationary_mean_current_error=current_error)


if __name__ == '__main__':
    main()
