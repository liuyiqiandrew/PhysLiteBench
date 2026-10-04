"""Validate piezoelectric-waves revision 1; regenerate calibration only with --generate."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['piezoelectric-waves']


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
    sigma = 6.0
    seed = 12117
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
    parser.add_argument('--output', type=Path, default=ROOT/'jobs/piezoelectric-waves-validation/summary.json')
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
    oracle=module(ROOT/'tasks/piezoelectric-waves/solution/model.py')
    es=[e for group in ref.hidden_inputs().values() for e in group]
    model=oracle_class();model.lame_parameter=ref.TRUE_PARAMETER
    reference_error=float(np.max(abs(model.predict(es)-ref.predict(es))))
    corner_error=0.;minimum_eigenvalue=float('inf');symmetry_error=0.
    minimum_difference_eigenvalue=float('inf');gauss_error=0.;curl_error=0.
    for parameter in [24.,32.,50.]:
        model.lame_parameter=parameter
        for theta in np.linspace(0,np.pi,13):
            for azimuth in [.0,.7,1.6]:
                e=ref.reading(theta,azimuth)
                n=np.array(e['direction']);polarization=np.einsum('kij,i->kj',oracle.PIEZOELECTRIC,n)
                g=n@polarization;eps=oracle.DIELECTRIC
                mechanical=oracle.SHEAR_MODULUS*np.eye(3)+(parameter*1e9+oracle.SHEAR_MODULUS)*np.outer(n,n)
                effective=mechanical+np.outer(g,g)/(n@eps@n)
                eig,vec=np.linalg.eigh(effective)
                minimum_eigenvalue=min(minimum_eigenvalue,float(min(eig)))
                corner_error=max(corner_error,float(np.max(abs(np.sqrt(eig/oracle.DENSITY)-ref.modes(n,parameter)))))
                for u in vec.T:
                    E=-n*(g@u)/(n@eps@n)
                    displacement=eps@E+polarization@u
                    gauss_error=max(gauss_error,float(abs(n@displacement)))
                    curl_error=max(curl_error,float(np.linalg.norm(np.cross(n,E))))
                difference=polarization.T@np.linalg.solve(eps,polarization)-np.outer(g,g)/(n@eps@n)
                minimum_difference_eigenvalue=min(minimum_difference_eigenvalue,float(min(np.linalg.eigvalsh(difference))))
                opposite=[dict(direction=(-n).tolist(),branch=b) for b in range(3)]
                direct=[dict(direction=n.tolist(),branch=b) for b in range(3)]
                symmetry_error=max(symmetry_error,float(np.max(abs(model.predict(opposite)-model.predict(direct)))))
    axial=ref.modes([0,0,1],32.)
    axial_expected=np.sqrt(np.array([20e9,20e9,72e9+18**2/1e-8])/6000.)
    axial_error=float(np.max(abs(axial-axial_expected)))
    assert reference_error<1e-8 and corner_error<1e-8 and axial_error<1e-8
    assert minimum_eigenvalue>0 and minimum_difference_eigenvalue>-1e-4
    assert symmetry_error<1e-8 and gauss_error<1e-10 and curl_error<1e-6
    return dict(saddle_reference_max_error=reference_error,parameter_direction_corner_error=corner_error,
                axial_analytic_error=axial_error,minimum_acoustic_stiffness_Pa=minimum_eigenvalue,
                gauss_residual=gauss_error,electric_curl_residual=curl_error,
                direction_reversal_error=symmetry_error,
                minimum_local_minus_bulk_stiffness_eigenvalue_Pa=minimum_difference_eigenvalue)


if __name__ == '__main__':
    main()
