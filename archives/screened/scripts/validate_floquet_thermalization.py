"""Validate floquet-thermalization revision 2; regenerate calibration only with --generate."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['floquet-thermalization']


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
    sigma = np.array([.002 if e['observable']=='population' else 2e-6 for e in inputs])
    seed = 14121
    if generate:
        noise = np.random.default_rng(seed).normal(0, sigma, len(inputs))
        records = [dict(input=e, value=float(v), sigma=float(s)) for e, v, s in zip(inputs, noiseless+noise, sigma)]
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
        sample = [dict(input=e, value=float(v), sigma=float(s)) for e, v, s in zip(inputs, noisy, sigma)]
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
    parser.add_argument('--output', type=Path, default=ROOT/'jobs/floquet-thermalization-r2-validation/summary.json')
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


def energy_kernel(module,amplitude,frequency,bath,horizon=240.,step=.02,terms=1536,phases=64):
    from scipy.integrate import simpson
    temperature=module.TEMPERATURES[bath];a=1/module.CUTOFFS[bath];strength=module.STRENGTHS[bath]
    times=np.linspace(0,horizon,round(horizon/step)+1)
    shifts=a+np.arange(1,terms+1)/temperature
    theta=np.arange(phases)*2*np.pi/phases
    down=np.empty(len(times));up=np.empty(len(times))
    for offset in range(0,len(times),256):
        t=times[offset:offset+256]
        z=shifts[None,:]+1j*t[:,None]
        kernel=strength/np.pi*((a+1j*t)**-3+np.sum(z**-3-z.conj()**-3,axis=1))
        modulation=np.mean(np.exp(1j*amplitude/frequency*(np.sin(theta)[None,:]-np.sin(theta[None,:]-frequency*t[:,None]))),axis=1)
        down[offset:offset+len(t)]=2*np.real(kernel*np.exp(1j*t)*modulation)
        up[offset:offset+len(t)]=2*np.real(kernel*np.exp(-1j*t)*modulation.conj())
    return np.array([simpson(down,x=times),simpson(up,x=times)])

def independent_checks(ref,oracle_class,baseline_class):
    oracle=module(ROOT/'tasks/floquet-thermalization/solution/model.py')
    baseline=module(ROOT/'scripts/floquet_thermalization_baseline.py')
    es=[e for group in ref.hidden_inputs().values() for e in group]
    c=ref.TRUE_PARAMETER
    truth=ref.predict(es,c)
    reference_error=float(np.max(abs(oracle.predict_at(es,c)-truth)))
    counting_refinement=float(np.max(abs(ref.predict(es,c,step=.00025)-truth)))
    fft_refinement=float(np.max(abs(ref.predict(es,c,samples=2048)-truth)))
    population=ref.population_inputs()
    population_error=float(np.max(abs(oracle.predict_at(population,c)-ref.predict(population,c))))
    population_shortcut_error=float(np.max(abs(oracle.predict_at(population,c)-baseline.predict_at(population,c))))
    kernel_errors=[];kernel_refinements=[]
    for bath in [0,1]:
        expected=np.array([oracle.coefficients(1.8,1.3)[i][bath] for i in [2,3]])
        coarse=energy_kernel(oracle,1.8,1.3,bath,horizon=120.,step=.04,terms=768)
        fine=energy_kernel(oracle,1.8,1.3,bath)
        kernel_errors.append(float(np.max(abs(fine-expected))))
        kernel_refinements.append(float(np.max(abs(fine-coarse))))
    # The static limit counts the actual fixed gap; total bath heat balances
    # the population energy change, with either initial preparation.
    static_error=0.;steady_entropy_min=float('inf');normalization_error=0.;corner_error=0.
    for amplitude in [0.,2.5]:
        for frequency in [.6,2.]:
            for c0 in [.002,.008]:
                experiments=[ref.experiment(amplitude,frequency,p,t,o,b) for p in [0.,1.] for t in [1,200] for o,b in [('population',0),('heat',0),('heat',1)]]
                corner_error=max(corner_error,float(np.max(abs(oracle.predict_at(experiments,c0)-ref.predict(experiments,c0)))))
                if amplitude==0:
                    for p in [0.,1.]:
                        pair=[ref.experiment(0.,frequency,p,200,'heat',b) for b in [0,1]]
                        pe=oracle.predict_at([ref.experiment(0.,frequency,p,200,'population',0)],c0)[0]
                        static_error=max(static_error,abs(sum(oracle.predict_at(pair,c0))-(p-pe)/(400*np.pi/frequency)))
            from scipy.special import jv
            normalization_error=max(normalization_error,abs(sum(jv(np.arange(-32,33),amplitude/frequency)**2)-1))
            down,up,qd,qu=oracle.coefficients(amplitude,frequency)
            p=np.sum(up)/np.sum(up+down)
            heats=qd*p+qu*(1-p)
            steady_entropy_min=min(steady_entropy_min,float(np.sum(heats/np.array(oracle.TEMPERATURES))))
    assert reference_error<1e-10 and counting_refinement<1e-10 and fft_refinement<1e-10
    assert population_error<1e-12 and population_shortcut_error<1e-12
    assert max(kernel_errors)<1e-7 and max(kernel_refinements)<3e-7
    assert static_error<1e-12 and normalization_error<1e-12 and corner_error<1e-9 and steady_entropy_min>=-1e-12
    return dict(tilted_energy_counting_error=reference_error,counting_step_refinement=counting_refinement,
                fft_1024_vs_2048_error=fft_refinement,population_reference_error=population_error,
                population_shortcut_agreement=population_shortcut_error,
                independent_time_domain_heat_coefficient_errors=kernel_errors,
                time_domain_refinement_errors=kernel_refinements,static_first_law_error=static_error,
                sideband_weight_normalization_error=normalization_error,
                parameter_and_input_corner_error=corner_error,minimum_stationary_bath_entropy_rate_per_coupling=steady_entropy_min)


if __name__ == '__main__':
    main()
