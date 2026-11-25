"""Validate floating-contact-noise revision 2; regenerate calibration only with --generate."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['floating-contact-noise']


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
    inputs = ref.calibration_inputs()*2
    noiseless = ref.predict(inputs, ref.TRUE_PARAMETER)
    sigma = .003*max(np.max(np.abs(noiseless)), 1e-10)
    seed = 20021
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
    parser.add_argument('--output', type=Path, default=ROOT/'jobs/floating-contact-noise-r2-validation/summary.json')
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
    impl=module(ROOT/'tasks/floating-contact-noise/solution/model.py')
    errors=[];calibration=[];refinement=[];probability_bounds=[];unitarity=[];odd=[];cgf_errors=[]
    for alpha in [.25,ref.TRUE_PARAMETER,.7]:
        oracle=oracle_class();oracle.mixing_angle=alpha
        base=baseline_class();base.mixing_angle=alpha
        for beta in np.linspace(.15,.8,27):
            es=[ref.experiment(v,beta,o,lead,other) for v in [-1.5,0.,1.5]
                for o,lead,other in [('current',0,0),('current',1,1),('noise',0,0),('noise',1,1),('noise',0,1),('third_cumulant',0,0),('third_cumulant',1,1)]]
            predicted=oracle.predict(es);truth=ref.predict(es,alpha)
            errors.append(float(np.max(abs(predicted-truth))))
            cal=[e for e in es if e['observable']!='third_cumulant']
            calibration.append(float(np.max(abs(oracle.predict(cal)-base.predict(cal)))))
            refinement.append(float(np.max(abs(ref.cumulants(alpha,beta,1024)-ref.cumulants(alpha,beta,2048)))))
            s,g,mu,current=impl.dc_state(ref.experiment(1.,beta),alpha)
            unitarity.append(float(np.max(abs(s@s.T-np.eye(3)))))
            phase=2*np.pi*(np.arange(1024)+.5)/1024;z=np.exp(1j*phase)
            transmission=abs(s[1,0]+s[1,2]*z*s[2,0]/(1-s[2,2]*z))**2
            probability_bounds.extend([float(transmission.min()),float(transmission.max())])
            # Cauchy derivatives of the analytic saddle CGF check the third
            # derivative algebra independently of the phase-average reference.
            t=s*s;count=.08*np.exp(1j*2*np.pi*np.arange(64)/64)
            factor=(t[0,0]+t[1,1])/2+t[0,1]*np.exp(count)+.5*np.sqrt((t[1,1]-t[0,0])**2+4*t[2,0]*t[2,1]*np.exp(count))
            derivative=6*np.mean(np.log(factor)*np.exp(-3j*2*np.pi*np.arange(64)/64)).real/.08**3
            cgf_errors.append(abs(derivative-impl.third_cumulant(s,g,mu,0)))
            odd.append(abs(oracle.predict([ref.experiment(1.,beta),ref.experiment(-1.,beta)]).sum()))
    assert max(errors)<2e-11 and max(refinement)<2e-11
    assert max(calibration)<1e-12 and max(unitarity)<1e-12 and max(odd)<1e-12
    assert min(probability_bounds)>-1e-12 and max(probability_bounds)<1+1e-12
    assert max(cgf_errors)<1e-9
    s,g,mu,current=impl.dc_state(ref.experiment(1.,1e-6),ref.TRUE_PARAMETER)
    transmission=np.sin(2*ref.TRUE_PARAMETER)**2
    disconnected=abs(impl.third_cumulant(s,g,mu,0)-transmission*(1-transmission)*(1-2*transmission))
    assert disconnected<1e-10
    grid=np.linspace(.25,.7,501);inputs=ref.calibration_inputs();truth=ref.predict(inputs,ref.TRUE_PARAMETER)
    loss=[np.mean((impl.predict_at(inputs,alpha)-truth)**2) for alpha in grid]
    minima=[i for i in range(1,len(grid)-1) if loss[i]<loss[i-1] and loss[i]<loss[i+1]]
    assert len(minima)==1 and abs(grid[minima[0]]-ref.TRUE_PARAMETER)<.001
    return dict(phase_average_reference_max_error=max(errors),phase_quadrature_refinement=max(refinement),
                shared_calibration_max_error=max(calibration),unitarity_error=max(unitarity),
                effective_transmission_min=min(probability_bounds),effective_transmission_max=max(probability_bounds),
                cauchy_third_derivative_error=max(cgf_errors),bias_reversal_error=max(odd),
                disconnected_bernoulli_third_error=disconnected,calibration_local_minima=len(minima))


if __name__ == '__main__':
    main()
