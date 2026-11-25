"""Validate elastohydrodynamic-film revision 2; regenerate calibration only with --generate."""
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
    sigma = np.full(len(inputs),.04)
    seed = 15124
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
    parser.add_argument('--output', type=Path, default=ROOT/'jobs/elastohydrodynamic-film-r2-validation/summary.json')
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
    from scipy.linalg import eigh
    from numpy.polynomial.legendre import Legendre
    oracle=module(ROOT/'tasks/elastohydrodynamic-film/solution/model.py')
    baseline=module(ROOT/'scripts/elastohydrodynamic_film_baseline.py')
    es=[e for group in ref.hidden_inputs().values() for e in group]
    truth=ref.predict(es,ref.TRUE_PARAMETER)
    exact=oracle.predict_at(es,ref.TRUE_PARAMETER)
    reference_error=float(np.max(abs(exact-truth)))
    selected=[group[0] for group in ref.hidden_inputs().values()]
    ref_coarse=ref.predict(selected,1.1)
    ref_fine=ref.predict(selected,1.1,points=41,degree=20)
    reference_refinement=float(np.max(abs(ref_coarse-ref_fine)))
    def prediction_at_points(experiments,modulus,points):
        out=[]
        for e in experiments:
            setup={k:v for k,v in e.items() if k not in ['pressure','elapsed_time']}
            rates,amplitudes=oracle.relaxation(json.dumps(setup,sort_keys=True),points)
            out.append(e['pressure']/modulus*np.sum(amplitudes*np.exp(-modulus*rates*e['elapsed_time'])))
        return np.array(out)
    spectral_refinement=float(np.max(abs(prediction_at_points(selected,1.1,65)-oracle.predict_at(selected,1.1))))
    cal=ref.calibration_inputs()
    calibration_equivalence=float(np.max(abs(oracle.predict_at(cal,1.1)-baseline.predict_at(cal,1.1))))
    # Evaluate displacement and stress at the actual clamped/loaded surfaces
    # from the independent variational bulk solution, not its surface kernel.
    e=ref.experiment(1.2,.7,2,.3,3,.2,2,.1,.2,thickness=1.2)
    points=41;degree=28;q=e['fundamental_wavenumber']*e['thickness']
    stiffness,surface=ref.elastic_problem(q,e['contrast'],e['pattern_mode'],e['pattern_phase'],points,degree)
    x,d=ref.grid(points);d=q*d;r=1+e['contrast']*np.cos(e['pattern_mode']*x+e['pattern_phase'])
    pressure=np.cos(e['pressure_mode']*x+e['pressure_phase'])
    displacement=np.linalg.solve(stiffness,-surface.T@pressure)
    u=displacement[:points*degree].reshape(degree,points);w=displacement[points*degree:].reshape(degree,points)
    derivative_top=1+np.arange(degree)*(np.arange(degree)+1)
    mu=1/(2*(1+.48));lam=.48/((1+.48)*(1-2*.48));longitudinal=lam+2*mu
    shear=mu*r*(derivative_top@u+d@np.sum(w,axis=0))
    normal=r*(lam*d@np.sum(u,axis=0)+longitudinal*(derivative_top@w))
    traction_error=float(max(np.max(abs(shear)),np.max(abs(normal+pressure))))
    weak_residual=float(np.max(abs(stiffness@displacement+surface.T@pressure)))
    bottom_basis=np.array([0.*Legendre.basis(j)(-1.) for j in range(degree)])
    clamp_error=float(max(np.max(abs(bottom_basis@u)),np.max(abs(bottom_basis@w))))
    # Reciprocity, positivity and liquid mass conservation for both closures.
    symmetry_error=0.;minimum_eigenvalue=float('inf');mass_error=0.;energy_error=0.;pressure_mean_change=[]
    for physical in [oracle,baseline]:
        x,modes,d=physical.grid(49)
        c=.001/1.1e6*physical.surface_compliance(1.,.7,1,.4)
        symmetry_error=max(symmetry_error,float(np.max(abs(c-c.T))))
        minimum_eigenvalue=min(minimum_eigenvalue,float(np.linalg.eigvalsh(c)[0]))
        mobility=(10e-6)**3/(12*.15);flow=-mobility*1e6*d@d
        rates,vectors=eigh(flow,c);rates=np.maximum(rates,0)
        p0=50*np.cos(x);p=(vectors*np.exp(-rates*.4))@(vectors.T@c@p0)
        p_dot=-np.linalg.solve(c,flow@p)
        mass_error=max(mass_error,abs(float(np.mean(c@p)-np.mean(c@p0)))*1e9)
        derivative=float(p@c@p_dot/len(x));dissipation=float(mobility*1e6*np.mean((d@p)**2))
        energy_error=max(energy_error,abs(derivative+dissipation))
        pressure_mean_change.append(float(np.mean(p)-np.mean(p0)))
    # Simultaneous shifts of material, preparation and detector leave the
    # measurement unchanged. A uniform pressure state is stationary.
    shift=.37;translated=[]
    for e0 in selected:
        moved=dict(e0)
        for name,mode in [('pattern_phase','pattern_mode'),('pressure_phase','pressure_mode'),('detector_phase','detector_mode')]:moved[name]+=moved[mode]*shift
        translated.append(moved)
    translation_error=float(np.max(abs(oracle.predict_at(selected,1.1)-oracle.predict_at(translated,1.1))))
    uniform=[ref.experiment(1.1,.7,2,.2,0,0.,m,.3,t) for m in [0,1,2,3] for t in [0.,2.]]
    values=oracle.predict_at(uniform,1.1).reshape(-1,2)
    stationary_error=float(np.max(abs(values[:,0]-values[:,1])))
    zero_inputs=[dict(e0,pressure=0.) for e0 in selected]
    zero_force_error=float(np.max(abs(oracle.predict_at(zero_inputs,1.1))))
    corners=[ref.experiment(1.2,.7,2,.3,3,.2,m,.1,t,thickness=1.2,gap=12.,pressure=60.) for m in [0,1,2,3,4,5] for t in [0.,.1,2.]]
    corner_error=0.
    for modulus in [.7,1.6]:corner_error=max(corner_error,float(np.max(abs(oracle.predict_at(corners,modulus)-ref.predict(corners,modulus,points=41,degree=20)))))
    assert reference_error<1e-4 and reference_refinement<1e-4 and spectral_refinement<1e-6
    assert calibration_equivalence<1e-8 and traction_error<1e-6 and weak_residual<1e-9 and clamp_error==0
    assert symmetry_error<1e-20 and minimum_eigenvalue>0 and mass_error<1e-8 and energy_error<1e-15
    assert translation_error<1e-6 and stationary_error<1e-8 and zero_force_error==0 and corner_error<1e-4
    return dict(variational_reference_max_error_nm=reference_error,reference33x16_to41x20_error_nm=reference_refinement,
                spectral49_to65_error_nm=spectral_refinement,homogeneous_full_wave_calibration_equivalence_nm=calibration_equivalence,
                top_traction_degree28_error=traction_error,weak_elastic_residual=weak_residual,bottom_clamp_error=clamp_error,
                reciprocity_error=symmetry_error,minimum_compliance_eigenvalue_m_per_pa=minimum_eigenvalue,
                both_closures_mass_error_nm=mass_error,both_closures_energy_dissipation_error=energy_error,
                both_closures_pressure_mean_change_pa=pressure_mean_change,translation_covariance_error_nm=translation_error,
                uniform_pressure_stationarity_error_nm=stationary_error,zero_force_error_nm=zero_force_error,
                parameter_and_input_corner_error_nm=corner_error)


if __name__=='__main__':main()
