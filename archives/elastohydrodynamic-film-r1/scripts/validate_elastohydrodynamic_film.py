"""Validate elastohydrodynamic-film revision 1; regenerate calibration only with --generate."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['elastohydrodynamic-film']


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
    sigma = np.full(len(inputs),.03)
    seed = 15123
    if generate:
        noise = np.random.default_rng(seed).normal(0, sigma, len(inputs))
        records = [dict(input=e, value=float(v), sigma=float(s)) for e, v, s in zip(inputs, noiseless+noise, sigma)]
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
    parser.add_argument('--output', type=Path, default=ROOT/'jobs/elastohydrodynamic-film-validation/summary.json')
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
    from scipy.linalg import expm
    oracle=module(ROOT/'tasks/elastohydrodynamic-film/solution/model.py')
    baseline=module(ROOT/'scripts/elastohydrodynamic_film_baseline.py')
    es=[e for group in ref.hidden_inputs().values() for e in group]
    exact=oracle.predict_at(es,ref.TRUE_PARAMETER)
    reference=ref.predict(es,ref.TRUE_PARAMETER)
    reference_error=float(np.max(abs(exact-reference)))
    refinement=float(np.max(abs(ref.predict(es,ref.TRUE_PARAMETER,basis_count=26)-reference)))
    transfer_error=0.;traction_error=0.;positive_compliance=float('inf')
    nu=.48;mu=1/(2*(1+nu));lam=nu/((1+nu)*(1-2*nu));longitudinal=lam+2*mu
    for q in [.001,.1,.5,1.,2.,4.,6.]:
        # [U,W,T,S] with ux=U sin(kx), uz=W cos(kx), shear=T sin,
        # normal stress=S cos. Solve bottom U=W=0, top T=0,S=-1.
        operator=np.array([[0,q,1/mu,0],[-lam*q/longitudinal,0,0,1/longitudinal],
                           [(longitudinal-lam**2/longitudinal)*q*q,0,0,lam*q/longitudinal],
                           [0,0,-q,0]])
        matrix=expm(operator)
        base_stress=np.linalg.solve(matrix[2:,2:],np.array([0.,-1.]))
        top=matrix[:,2:]@base_stress
        analytic=oracle.compliance(q,1.,1.)*1e9
        transfer_error=max(transfer_error,abs(analytic+top[1]))
        traction_error=max(traction_error,float(np.max(abs(top[2:]-np.array([0.,-1.])))))
        positive_compliance=min(positive_compliance,ref.elastic_energy_compliance(q))
    uniform_error=abs(oracle.compliance(1e-7,1.,1.)/oracle.compliance(0.,1.,1.)-1)
    halfspace=2*(1-nu**2)/(1e6*20000)
    halfspace_error=abs(oracle.compliance(20.,1.,1.)/halfspace-1)
    calibration_equivalence=float(np.max(abs(oracle.predict_at(ref.calibration_inputs(),1.1)-baseline.predict_at(ref.calibration_inputs(),1.1))))
    corner_error=0.;max_deformation_ratio=0.;dissipation_error=0.
    for modulus in [.7,1.6]:
        corners=[ref.experiment(k,d,h,t,100.) for k in [0.,.2,4.] for d in [.5,1.5] for h in [5.,15.] for t in [0.,.1]]
        corner_error=max(corner_error,float(np.max(abs(oracle.predict_at(corners,modulus)-ref.predict(corners,modulus,basis_count=26)))))
        for e in corners:
            c=oracle.compliance(e['wavenumber'],e['thickness'],modulus);k=e['wavenumber']*1000;gap=e['gap']*1e-6
            max_deformation_ratio=max(max_deformation_ratio,e['pressure']*c/gap)
            mobility=gap**3/(12*.15);rate=mobility*k*k/c
            energy_loss=.5*c*e['pressure']**2*rate
            dissipation=.5*mobility*k*k*e['pressure']**2
            dissipation_error=max(dissipation_error,abs(energy_loss-dissipation))
    # Dense compliance scan bounds deformation over the full permitted domain;
    # the largest pressure/depth and smallest modulus/gap are the extremes.
    max_deformation_ratio=max(100*oracle.compliance(float(k),1.5,.7)/(5e-6) for k in np.linspace(0,4,1001))
    # Independent periodic finite-volume flux symbol converges to the stated
    # continuum lubrication balance. Check the measured displacement itself.
    e=ref.experiment(1.3,1.2,10.,.3);c=oracle.compliance(e['wavenumber'],e['thickness'],1.1)
    k=e['wavenumber']*1000;mobility=(e['gap']*1e-6)**3/(12*.15)
    continuum=oracle.predict_at([e],1.1)[0];fv_errors=[]
    for cells in [128,256]:
        spacing=2*np.pi/k/cells
        flux_eigenvalue=4*np.sin(np.pi/cells)**2/spacing**2
        prediction=e['pressure']*c*np.exp(-mobility*flux_eigenvalue*e['elapsed_time']/c)*1e9
        fv_errors.append(abs(prediction-continuum))
    assert reference_error<1e-7 and refinement<1e-7
    assert transfer_error<1e-10 and traction_error<1e-10 and positive_compliance>0
    assert uniform_error<1e-10 and halfspace_error<1e-10 and calibration_equivalence<1e-12
    assert corner_error<1e-7 and max_deformation_ratio<.025 and dissipation_error<1e-18
    assert 3.9<fv_errors[0]/fv_errors[1]<4.1
    return dict(energy_reference_max_error_nm=reference_error,basis18_vs26_error_nm=refinement,
                independent_stress_transfer_compliance_error=transfer_error,traction_residual=traction_error,
                minimum_dimensionless_compliance=positive_compliance,uniform_limit_relative_error=uniform_error,
                halfspace_limit_relative_error=halfspace_error,calibration_equivalence=calibration_equivalence,
                parameter_input_corner_error_nm=corner_error,maximum_gap_deformation_fraction=max_deformation_ratio,
                elastic_energy_fluid_dissipation_error=dissipation_error,finite_volume128_256_errors_nm=fv_errors)


if __name__=='__main__':main()
