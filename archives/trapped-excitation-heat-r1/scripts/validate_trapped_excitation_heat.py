"""Validate trapped-excitation-heat revision 1; regenerate calibration only with --generate."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['trapped-excitation-heat']


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
    sigma = .006*max(np.max(np.abs(noiseless)), 1e-10)
    seed = 15111
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
    parser.add_argument('--output', type=Path, default=ROOT/'jobs/trapped-excitation-heat-validation/summary.json')
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
    impl=module(ROOT/'tasks/trapped-excitation-heat/solution/model.py')
    errors=[];refinement=[];mass=[];energy=[];entropy_rises=[]
    for es in ref.hidden_inputs().values():
        exact=oracle_class();exact.diffusivity=ref.TRUE_PARAMETER
        coarse=ref.predict(es,ref.TRUE_PARAMETER,points=128)
        fine=ref.predict(es,ref.TRUE_PARAMETER,points=512)
        scale=scale_for('',es,fine)
        errors.append(float(np.max(abs(exact.predict(es)-fine))/scale))
        refinement.append(float(np.max(abs(coarse-fine))/scale))
    for e in [ref.experiment(),ref.experiment(temperature=.85,thermal_amplitude=-.07,mode=3)]:
        fields,u=ref.profile(e,[0.,.2,1.,4.,8.],ref.TRUE_PARAMETER,points=256)
        c,t=fields;free=c/(1+np.exp(2/t));bound=c-free
        entropy=np.mean(.15*np.log(t)-bound*np.log(bound)-free*np.log(free)+c,axis=0)
        mass.append(float(np.ptp(np.mean(c,axis=0))))
        energy.append(float(np.ptp(np.mean(u,axis=0))))
        entropy_rises.append(float(np.min(np.diff(entropy))))
    assert max(errors)<.002 and max(refinement)<.006
    assert max(mass+energy)<1e-10 and min(entropy_rises)>-1e-9
    # Mobile-energy flux fixes the sign of the initial thermal response.
    e=ref.experiment(time=.002);a=oracle_class();a.diffusivity=ref.TRUE_PARAMETER
    b=baseline_class();b.diffusivity=ref.TRUE_PARAMETER
    true=float(a.predict([e])[0]);wrong=float(b.predict([e])[0])
    assert true<0<wrong
    return dict(spectral_fv_relative_max=max(errors),fv_refinement_relative_max=max(refinement),mass_drift=max(mass),energy_drift=max(energy),minimum_entropy_increase=min(entropy_rises),early_temperature_coefficients=dict(oracle=true,shortcut=wrong))


if __name__ == '__main__':
    main()
