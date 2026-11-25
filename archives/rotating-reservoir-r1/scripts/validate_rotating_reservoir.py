"""Validate rotating-reservoir calorimetry and laboratory mechanical work."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from scipy.integrate import quad

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/rotating-reservoir'


def module(path):
    spec=importlib.util.spec_from_file_location('validation_'+str(abs(hash(str(path)))),path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def physical_checks(good,bad,ref):
    from scipy.integrate import quad_vec
    R=good.R;b=good.B_DRAG;identity=np.eye(2)
    hidden=sum(ref.hidden_inputs().values(),[]);cal=ref.calibration_inputs()
    ref.covariance.cache_clear();start=time.perf_counter();truth=ref.predict(hidden);runtime=time.perf_counter()-start
    agreement=float(max(abs(good.predict_at(hidden,.67)-truth)))
    cal_equivalence=float(max(abs(good.predict_at(cal,.67)-bad.predict_at(cal,.67))))
    covariance_error=[];balance=[];heat_identity=[];entropy=[];decay=[];covmin=[];mirror=[];quadrature=[]
    for drag in [.4,.67,1.1]:
        for omega in [-.65,0.,.65]:
            for kx,ky in [(.8,2.),(.8,3.2),(1.8,2.),(1.8,3.2)]:
                for Ta,Tb in [(.6,.6),(.6,1.4),(1.6,.6),(1.6,1.4)]:
                    e=ref.experiment(omega,kx,ky,Ta,Tb);C=good.stationary_covariance(e,drag);A,Q=ref.matrices(e,drag)
                    covariance_error.append(float(max(abs(A@C+C@A.T+Q@Q.T).ravel())))
                    decay.append(float(max(np.linalg.eigvals(A).real)));covmin.append(float(min(np.linalg.eigvalsh(C))))
                    x2=np.trace(C[:2,:2]);v2=np.trace(C[2:,2:]);L=np.trace(R@C[:2,2:]);tau=-b*(L-omega*x2)
                    q1=drag*(v2-2*Ta);q2=good.heat_rate(e,drag);lab=bad.heat_rate(e,drag)
                    balance.append(abs(q1+q2-omega*tau));heat_identity.append(abs(q2-lab-omega*tau));entropy.append(q1/Ta+q2/Tb)
                    mirror.append(abs(q2-good.heat_rate(dict(e,angular_speed=-omega),drag)))
    # Third derivation: the susceptibility spectrum integrates to the state covariance.
    for drag in [.4,1.1]:
        for omega in [-.65,.4]:
            for kx,ky,Ta,Tb in [(.8,3.2,.6,1.4),(1.8,2.,1.6,.6)]:
                e=ref.experiment(omega,kx,ky,Ta,Tb);K=np.diag([kx,ky])
                def spectrum(frequency):
                    G=np.linalg.inv(K-b*omega*R-(frequency**2+1j*frequency*(drag+b))*identity)
                    response=np.vstack([G,-1j*frequency*G])
                    return 2*(drag*Ta+b*Tb)/np.pi*np.real(response@response.conj().T)
                spectral,error=quad_vec(spectrum,0,np.inf,epsabs=1e-11,epsrel=1e-11)
                impulse=ref.covariance(**e,drag=drag);C=good.stationary_covariance(e,drag)
                quadrature.extend([float(max(abs(spectral-impulse).ravel())),float(max(abs(spectral-C).ravel()))])
    limits=[];equilibrium=[]
    for omega in [-.65,.35,.65]:
        k=1.3;T=1.1;e=ref.experiment(omega,k,k,2.,T);C=good.stationary_covariance(e,0.);s=T/(k-omega**2)
        expected=np.block([[s*identity,-omega*s*R],[omega*s*R,(T+omega**2*s)*identity]])
        torque=-b*(np.trace(R@C[:2,2:])-omega*np.trace(C[:2,:2]))
        limits.extend([float(max(abs(C-expected).ravel())),abs(good.heat_rate(e,0.)),abs(torque)])
    for kx,ky,T in [(.8,3.2,.6),(1.8,2.,1.4)]:
        e=ref.experiment(0.,kx,ky,T,T);C=good.stationary_covariance(e,.67)
        equilibrium.extend([float(max(abs(C-np.diag([T/kx,T/ky,T,T])).ravel())),abs(good.heat_rate(e,.67))])
    fits=[];conductance_errors=[]
    for drag in [.401,.67,1.099]:
        exact=good.predict_at(cal,drag);records=[dict(input=e,value=float(y),sigma=.001) for e,y in zip(cal,exact)]
        fits.append(abs(good.Model().fit(records).drag/drag-1))
        expected=np.array([2*b*drag/(drag+b)*(e['temperature_a']-e['temperature_b']) for e in cal])
        conductance_errors.append(float(max(abs(exact-expected))))
    # Routh-Hurwitz stability margin of the two-dimensional damped oscillator.
    stability_bound=(.4+b)**2*(.8+2.)/2-(b*.65)**2
    assert stability_bound>0 and max(decay)<0 and min(covmin)>0
    for e in cal+hidden:
        assert -.65<=e['angular_speed']<=.65 and .8<=e['stiffness_x']<=1.8 and 2<=e['stiffness_y']<=3.2
        assert .6<=e['temperature_a']<=1.6 and .6<=e['temperature_b']<=1.4
    assert max([agreement]+covariance_error+balance+heat_identity+mirror+quadrature+limits+equilibrium+conductance_errors)<1e-8
    assert min(entropy)>-1e-10 and cal_equivalence==0. and max(fits)<1e-10 and runtime<45
    return {'oracle_impulse_reference_hidden_max':agreement,'cold_hidden_reference_seconds':runtime,
        'calibration_control_difference':cal_equivalence,'Lyapunov_balance_residual':max(covariance_error),
        'minimum_full_domain_covariance_eigenvalue':min(covmin),'maximum_full_domain_drift_real_eigenvalue':max(decay),
        'analytic_full_domain_stability_margin_lower_bound':stability_bound,'steady_first_law_residual':float(max(balance)),
        'heat_lab_power_motor_identity_error':float(max(heat_identity)),'minimum_entropy_production':float(min(entropy)),
        'rotation_reversal_heat_symmetry_error':float(max(mirror)),'independent_frequency_impulse_covariance_error':max(quadrature),
        'single_rotating_bath_Gibbs_heat_torque_covariance_error':float(max(limits)),'static_equal_temperature_equilibrium_error':float(max(equilibrium)),
        'calibration_conductance_formula_error':max(conductance_errors),'noiseless_fit_relative_errors':fits,
        'positive_conductance_derivative_full_drag_interval':True,'public_input_domain_verified':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--generate',action='store_true');parser.add_argument('--noise-trials',type=int,default=256)
    args=parser.parse_args();start=time.time()
    good=module(TASK/'solution/model.py');bad=module(ROOT/'scripts/rotating_reservoir_baseline.py');ref=module(TASK/'tests/reference.py')
    metadata=json.loads((TASK/'tests/metadata.json').read_text());true=ref.TRUE_PARAMETER;sigma=metadata['measurement_sigma'];limit=metadata['prediction_limit']
    inputs=ref.calibration_inputs();exact=ref.predict(inputs,true)
    if args.generate:
        values=exact+np.random.default_rng(metadata['calibration_seed']).normal(0,sigma,len(inputs))
        records=[dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,values)]
        for directory in ['environment','tests']:(TASK/directory/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records==json.loads((TASK/'tests/data/calibration.json').read_text()) and [r['input'] for r in records]==inputs
    assert all(r['sigma']==sigma for r in records)
    hidden=ref.hidden_inputs();truth={name:ref.predict(es,true) for name,es in hidden.items()}
    def chi(model,rs):
        residual=(model.predict(inputs)-np.array([r['value'] for r in rs]))/sigma
        return float(residual@residual/(len(rs)-1))
    def scores(model):return {name:float(np.sqrt(np.mean((model.predict(es)-truth[name])**2)/np.mean(truth[name]**2))) for name,es in hidden.items()}
    report={'revision':1,'calibration_seed':metadata['calibration_seed'],'noise_seed':metadata['noise_seed'],'noise_trials':args.noise_trials,'controls':{}}
    for label,source in [('oracle',good),('shortcut',bad)]:
        model=source.Model().fit(records);result={'parameter':model.drag,'parameter_relative_error':abs(model.drag/true-1),
            'calibration_chi2':chi(model,records),'hidden':scores(model)}
        assert result['calibration_chi2']<1.5 and result['parameter_relative_error']<.03
        assert (max(result['hidden'].values())<limit if label=='oracle' else min(result['hidden'].values())>limit)
        report['controls'][label]=result
    rng=np.random.default_rng(metadata['noise_seed']);parameters=[];chi2s=[];noise_oracle=[];noise_shortcut=[]
    for _ in range(args.noise_trials):
        sample=[dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,exact+rng.normal(0,sigma,len(inputs)))]
        a=good.Model().fit(sample);b=bad.Model().fit(sample);assert abs(a.drag-b.drag)<1e-10
        parameters.append(a.drag);chi2s.append(chi(a,sample))
        noise_oracle.append(max(scores(a).values()));noise_shortcut.append(min(scores(b).values()))
        assert noise_oracle[-1]<limit and noise_shortcut[-1]>limit
    assert max(chi2s)<1.5 and max(abs(np.array(parameters)/true-1))<.03
    report['noise']={'parameter_min':min(parameters),'parameter_max':max(parameters),'parameter_relative_error_max':float(max(abs(np.array(parameters)/true-1))),
        'calibration_chi2_max':max(chi2s),'calibration_pass_fraction':1.,'oracle_hidden_passes':args.noise_trials,'shortcut_hidden_passes':0,
        'oracle_hidden_max':max(noise_oracle),'shortcut_hidden_min':min(noise_shortcut)}
    extrema={}
    for label,source in [('oracle',good),('shortcut',bad)]:
        errors=[]
        for parameter in [min(parameters),max(parameters)]:
            model=source.Model();model.drag=parameter;errors.extend(scores(model).values())
        extrema[label]={'min':min(errors),'max':max(errors)}
    assert extrema['oracle']['max']<limit and extrema['shortcut']['min']>limit
    report['hidden_parameter_extrema_check']=extrema;report['physical_checks']=physical_checks(good,bad,ref);report['seconds']=time.time()-start
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TASK/'instruction.md',TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',TASK/'tests/metadata.json',TASK/'environment/data/calibration.json',ROOT/'scripts/rotating_reservoir_baseline.py',Path(__file__)]}
    output=ROOT/'jobs/rotating-reservoir-validation';output.mkdir(parents=True,exist_ok=True);(output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'results/rotating-reservoir-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('rotating-reservoir PASS',output/'summary.json',report['seconds'])
    print(json.dumps(report['controls'],indent=2));print(json.dumps(report['physical_checks'],indent=2))


if __name__=='__main__':main()
