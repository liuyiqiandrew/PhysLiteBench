"""Validate phonon-transport revision 2; regenerate calibration only with --generate."""
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
    seed = 13120
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
    parser.add_argument('--output', type=Path, default=ROOT/'jobs/phonon-transport-r2-validation/summary.json')
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
    es=[e for group in ref.hidden_inputs().values() for e in group];r=ref.TRUE_PARAMETER
    truth=ref.predict(es,r);error=float(np.max(abs(oracle.predict_at(es,r)-truth)))
    quadrature=float(np.max(abs(ref.predict(es,r,order=100)-truth)))
    truncation=float(np.max(abs(oracle.predict_at(es,r,order=80)-oracle.predict_at(es,r))))
    mu,w=ref.angles(80);c=ref.CAPACITIES;v=ref.SPEEDS
    energy=np.r_[c[0]*w,c[1]*w];momentum=np.r_[c[0]*w*mu/v[0],c[1]*w*mu/v[1]]
    normal=ref.generator(0.,1.,0.,80)
    energy_error=float(np.max(abs(energy@normal)));momentum_error=float(np.max(abs(momentum@normal)))
    weights=np.r_[c[0]*w,c[1]*w];weighted=np.diag(np.sqrt(weights))@normal@np.diag(1/np.sqrt(weights))
    max_eigen=float(np.max(np.linalg.eigvalsh(weighted.real)))
    # The source also conserves total energy and momentum and is passive. Its
    # extra conserved branch drift, not a conservation/sign mistake, is tested.
    order=64;source=baseline.operators(0.,1.,0.,order)
    energy_m=np.zeros(2*order);energy_m[[0,order]]=c
    momentum_m=np.zeros(2*order);momentum_m[[1,order+1]]=c/v/np.sqrt(3)
    source_conservation=max(float(np.max(abs(energy_m@source))),float(np.max(abs(momentum_m@source))))
    source_weights=np.repeat(c,order);weighted_source=np.diag(np.sqrt(source_weights))@source@np.diag(1/np.sqrt(source_weights))
    source_max=float(np.max(np.linalg.eigvalsh(weighted_source.real)))
    source_nullity=int(np.sum(abs(np.linalg.eigvalsh(weighted_source.real))<1e-10))
    physical_nullity=int(np.sum(abs(np.linalg.eigvalsh(weighted.real))<1e-10))
    assert max(error,quadrature,truncation,energy_error,momentum_error,source_conservation)<1e-10
    assert max(max_eigen,source_max)<1e-10 and source_nullity==3 and physical_nullity==2
    counter=ref.hidden_inputs()['counterflow'];expected=np.array([e['initial'][0][1]/3*np.exp(-(r+e['normal_rate'])*e['time']) for e in counter])
    counter_error=float(np.max(abs(oracle.predict_at(counter,r)-expected)));assert counter_error<1e-12
    return dict(angular_reference_error=error,angular_80_vs_100_error=quadrature,legendre_64_vs_80_error=truncation,normal_energy_residual=energy_error,normal_momentum_residual=momentum_error,normal_max_weighted_eigenvalue=max_eigen,source_conservation_error=source_conservation,source_max_weighted_eigenvalue=source_max,physical_normal_nullity=physical_nullity,shortcut_normal_nullity=source_nullity,zero_total_momentum_counterflow_error=counter_error)


if __name__ == '__main__':
    main()
