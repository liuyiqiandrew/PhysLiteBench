"""Validate coupled-oscillator-equilibrium revision 1; regenerate calibration only with --generate."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['coupled-oscillator-equilibrium']


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
    seed = 9328
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
    parser.add_argument('--output', type=Path, default=ROOT/'jobs/coupled-oscillator-equilibrium-validation/summary.json')
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


def fock_moments(spring,coupling,temperature,force,cutoff):
    # Galerkin matrix elements in the two uncoupled oscillator number bases.
    # Quadratic observables retain their exact matrix elements at the top level.
    frequencies = [np.sqrt(spring),1.6]
    number = np.arange(cutoff)
    lowering = np.diag(np.sqrt(np.arange(1,cutoff)),1)
    raising = lowering.T
    identity = np.eye(cutoff)
    positions = [(lowering+raising)/np.sqrt(2*w) for w in frequencies]
    probe = np.kron(positions[0],identity)
    diagonal = (frequencies[0]*(number[:,None]+.5)+frequencies[1]*(number[None,:]+.5)).reshape(-1)
    hamiltonian = np.diag(diagonal)+coupling*np.kron(positions[0],positions[1])-force*probe
    energies,vectors = np.linalg.eigh(hamiltonian)
    weights = np.exp(-(energies-energies[0])/temperature)
    weights /= sum(weights)
    def expectation(operator):
        return float(np.sum(weights*np.sum(vectors*(operator@vectors),axis=0)))
    mean = expectation(probe)
    quadratic = lowering@lowering+raising@raising
    q2 = np.kron((np.diag(2*number+1)+quadratic)/(2*frequencies[0]),identity)
    p2 = np.kron((np.diag(2*number+1)-quadratic)*frequencies[0]/2,identity)
    return np.array([mean,expectation(q2)-mean**2,expectation(p2)])


def independent_checks(ref, oracle_class, baseline_class):
    model = oracle_class(); model.spring = ref.TRUE_PARAMETER
    baseline = baseline_class(); baseline.spring = model.spring
    all_es = [e for es in ref.hidden_inputs().values() for e in es]
    residue_error = float(np.max(abs(model.predict(all_es)-ref.predict(all_es,model.spring))))
    assert residue_error < 1e-12
    fock_coarse_error = 0.
    fock_fine_error = 0.
    fock_cutoff_change = 0.
    for coupling,temperature,force in [(1.3,.08,.3),(-1.2,.3,-.4),(1.3,.5,.0)]:
        experiments = [dict(coupling=coupling,temperature=temperature,force=force,observable=obs)
                       for obs in ['mean_position','position_variance','momentum_variance']]
        physical = model.predict(experiments)
        coarse = fock_moments(model.spring,coupling,temperature,force,26)
        fine = fock_moments(model.spring,coupling,temperature,force,36)
        fock_coarse_error = max(fock_coarse_error,float(np.max(abs(coarse-physical))))
        fock_fine_error = max(fock_fine_error,float(np.max(abs(fine-physical))))
        fock_cutoff_change = max(fock_cutoff_change,float(np.max(abs(fine-coarse))))
    assert fock_fine_error < 2e-7 and fock_fine_error < fock_coarse_error/10
    zero_coupling = [dict(e,coupling=0.) for e in all_es]
    zero_coupling_error = float(np.max(abs(model.predict(zero_coupling)-baseline.predict(zero_coupling))))
    assert zero_coupling_error < 1e-12
    high_temperature_error = 0.
    minimum_uncertainty_product = float('inf')
    minimum_squared_frequency = float('inf')
    for spring in [.8,1.05,1.5]:
        model.spring = spring
        for coupling in [-1.3,0.,1.3]:
            minimum_squared_frequency = min(minimum_squared_frequency,float(np.linalg.eigvalsh([[spring,coupling],[coupling,2.56]])[0]))
            for temperature in [.08,.3,1.5]:
                es = [dict(force=.5,coupling=coupling,temperature=temperature,observable=obs)
                      for obs in ['position_variance','momentum_variance']]
                variances = model.predict(es)
                minimum_uncertainty_product = min(minimum_uncertainty_product,float(np.prod(variances)))
                assert np.max(abs(variances-ref.predict(es,spring))) < 1e-12
            es = [dict(e,temperature=1e4) for e in es]
            effective = spring-coupling**2/2.56
            classical = np.array([1e4/effective,1e4])
            high_temperature_error = max(high_temperature_error,float(np.max(abs(model.predict(es)/classical-1))))
    assert minimum_squared_frequency > 0 and minimum_uncertainty_product >= .25-1e-12
    assert high_temperature_error < 2e-8
    model.spring = ref.TRUE_PARAMETER
    ground = [dict(force=0.,coupling=1.3,temperature=1e-4,observable=obs)
              for obs in ['position_variance','momentum_variance']]
    ground_variances = model.predict(ground)
    reduced_ground_purity = float(1/(2*np.sqrt(np.prod(ground_variances))))
    assert reduced_ground_purity < .95
    # Static susceptibility obtained from both normal-mode residues equals the
    # Schur complement even though finite-frequency thermal fluctuations differ.
    eigenvalues,modes = np.linalg.eigh([[model.spring,1.3],[1.3,2.56]])
    susceptibility_error = float(abs(np.sum(modes[0]**2/eigenvalues)-1/(model.spring-1.3**2/2.56)))
    assert susceptibility_error < 1e-12
    return dict(green_function_residue_error=residue_error, fock_cutoff26_error=fock_coarse_error,
                fock_cutoff36_error=fock_fine_error, fock_cutoff_change=fock_cutoff_change,
                zero_coupling_error=zero_coupling_error, high_temperature_relative_error=high_temperature_error,
                minimum_uncertainty_product=minimum_uncertainty_product,
                minimum_squared_normal_frequency=minimum_squared_frequency,
                reduced_ground_state_purity=reduced_ground_purity,
                static_susceptibility_error=susceptibility_error)


if __name__ == '__main__':
    main()
