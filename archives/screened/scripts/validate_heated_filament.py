"""Validate heated-filament r1; calibration regeneration requires --generate."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]


def module(path):
    spec=importlib.util.spec_from_file_location('candidate_'+str(abs(hash(str(path)))),path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def score(prediction, truth):
    return float(np.sqrt(np.mean((prediction-truth)**2)/np.mean(truth**2)))


def metrics(model, records):
    prediction=model.predict([r['input'] for r in records])
    residual=(prediction-np.array([r['value'] for r in records]))/np.array([r['sigma'] for r in records])
    return dict(parameter=model.friction,parameter_relative_error=abs(model.friction/1.1-1),calibration_chi2=float(residual@residual)/(len(records)-1))


def green_spectrum(e, gamma):
    nodes,weights=np.polynomial.legendre.leggauss(64)
    x=(nodes+1)*np.pi/2; outer=weights*np.pi/2
    z=np.sqrt(1j*gamma*e['frequency'])
    chi=[]
    for position in x:
        response=0j
        for left,right in [(0.,position),(position,np.pi)]:
            y=left+(nodes+1)*(right-left)/2
            measure=weights*(right-left)/2
            lo=np.minimum(position,y);hi=np.maximum(position,y)
            if e['frequency']==0:
                green=lo*(np.pi-hi)/np.pi
            else:
                green=np.sinh(z*lo)*np.sinh(z*(np.pi-hi))/(z*np.sinh(z*np.pi))
            sensor=np.sqrt(2/np.pi)*sum(a*np.sin((n+1)*y) for n,a in enumerate(e['weights']))
            response+=np.sum(measure*green*sensor)
        chi.append(response)
    temperature=e['temperature']*(1+e['contrast']*np.cos(e['wavenumber']*x+e['phase']))
    return float(2*gamma*np.sum(outer*temperature*abs(np.array(chi))**2))


def independent_checks(ref, oracle, baseline):
    es=sum(ref.hidden_inputs().values(),[])
    gamma=ref.TRUE_PARAMETER
    a=oracle.Model();a.friction=gamma
    b=baseline.Model();b.friction=gamma
    truth=ref.predict(es,gamma)
    finer=ref.predict(es,gamma,intervals=1024)
    error=float(np.max(abs(a.predict(es)-truth)))
    refinement=float(np.max(abs(finer-truth)))
    assert error<1e-7 and refinement<1e-7
    anchors=ref.hidden_inputs()['single_mode_anchors']
    anchor_error=float(np.max(abs(a.predict(anchors)-b.predict(anchors))))
    uniform_error=0.;reflection_error=0.;green_error=0.;corner_error=0.;minimum_temperature_eigenvalue=1.
    for friction in [.6,1.1,1.8]:
        a.friction=b.friction=friction
        for omega in [0.,.45,4.,40.]:
            for wave in [1,4,8]:
                e=ref.experiment(ref.pair(1,6,-1),frequency=omega,temperature=.04,contrast=.85,wavenumber=wave,phase=.9)
                value=a.predict([e])[0]
                green_error=max(green_error,abs(value-green_spectrum(e,friction)))
                corner_error=max(corner_error,abs(value-ref.predict([e],friction)[0]))
                c=oracle.modal_temperatures(e['temperature'],e['contrast'],wave,e['phase'])
                minimum_temperature_eigenvalue=min(minimum_temperature_eigenvalue,float(np.linalg.eigvalsh(c).min()))
                reflected=dict(e,weights=(np.array(e['weights'])*(-1.)**np.arange(2,8)).tolist(),phase=float((-e['phase']-wave*np.pi+np.pi)%(2*np.pi)-np.pi))
                reflection_error=max(reflection_error,abs(value-a.predict([reflected])[0]))
                flat=dict(e,contrast=0.)
                n=np.arange(1,7);w=np.array(e['weights'])
                exact=2*friction*e['temperature']*np.sum(w*w/(n**4+(friction*omega)**2))
                uniform_error=max(uniform_error,abs(exact-a.predict([flat])[0]))
                anchor_error=max(anchor_error,float(np.max(abs(a.predict(anchors)-b.predict(anchors)))))
    # Integrate the two-sided spectrum and compare with the stationary Lyapunov covariance.
    a.friction=gamma
    e=ref.experiment(ref.pair(1,3,-1),temperature=.025,contrast=.8,wavenumber=2,phase=.4)
    covariance=oracle.modal_temperatures(e['temperature'],e['contrast'],e['wavenumber'],e['phase'])
    n=np.arange(1,7);w=np.array(e['weights'])
    stationary=2*covariance/(n[:,None]**2+n[None,:]**2)
    exact_variance=float(w@stationary@w)
    integral=quad(lambda omega: a.predict([dict(e,frequency=omega)])[0]/np.pi,0,np.inf,epsabs=1e-11)[0]
    variance_error=abs(exact_variance-integral)
    assert max(anchor_error,uniform_error,reflection_error,green_error,variance_error)<1e-10
    assert minimum_temperature_eigenvalue>0 and corner_error<1e-7
    # All calibration response derivatives are positive over the parameter interval.
    cal=ref.calibration_inputs();omega=np.array([e['frequency'] for e in cal])
    assert float(1-(1.8*omega.max())**2)>0
    return dict(reference_max_absolute_error=error,fd_refinement_max_absolute_error=refinement,
                dirichlet_green_max_absolute_error=float(green_error),allowed_corner_reference_error=float(corner_error),
                single_mode_closure_error=anchor_error,uniform_bath_identity_error=float(uniform_error),
                reflection_identity_error=float(reflection_error),integrated_spectrum_variance_error=float(variance_error),
                minimum_modal_temperature_eigenvalue=minimum_temperature_eigenvalue,
                calibration_monotonicity_margin=float(1-(1.8*omega.max())**2))


def validate(generate=False,noise_trials=256):
    task=ROOT/'tasks/heated-filament'
    ref=module(task/'tests/reference.py');oracle=module(task/'solution/model.py');baseline=module(ROOT/'scripts/heated_filament_baseline.py')
    inputs=ref.calibration_inputs()*4
    clean=ref.predict(inputs,ref.TRUE_PARAMETER)
    sigma=.008*clean
    seed=20371
    if generate:
        observed=clean+np.random.default_rng(seed).normal(0,sigma)
        records=[dict(input=e,value=float(v),sigma=float(s)) for e,v,s in zip(inputs,observed,sigma)]
        for part in ['environment','tests']:
            (task/part/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
    records=json.loads((task/'environment/data/calibration.json').read_text())
    assert records==json.loads((task/'tests/data/calibration.json').read_text())
    hidden=ref.hidden_inputs();truth={k:ref.predict(es,ref.TRUE_PARAMETER) for k,es in hidden.items()}
    report={'revision':1,'seed':seed,'noise_trials':noise_trials,'controls':{}}
    for label,m in [('oracle',oracle),('shortcut',baseline)]:
        exact=m.Model();exact.friction=ref.TRUE_PARAMETER
        discrepancy=float(np.max(abs(exact.predict(inputs)-clean)))
        assert discrepancy<1e-7
        model=m.Model().fit(records);result=metrics(model,records)
        result['calibration_reference_max_error']=discrepancy
        result['hidden']={k:score(model.predict(es),truth[k]) for k,es in hidden.items()}
        assert result['calibration_chi2']<1.5 and result['parameter_relative_error']<.03
        if label=='oracle':assert max(result['hidden'].values())<.025
        else:
            assert result['hidden']['single_mode_anchors']<.025
            assert min(v for k,v in result['hidden'].items() if k!='single_mode_anchors')>.025
        report['controls'][label]=result
    rng=np.random.default_rng(seed+10000);parameters=[];chi2=[]
    for index in range(noise_trials):
        sample=[dict(input=e,value=float(v),sigma=float(s)) for e,v,s in zip(inputs,clean+rng.normal(0,sigma),sigma)]
        model=oracle.Model().fit(sample);result=metrics(model,sample)
        parameters.append(model.friction);chi2.append(result['calibration_chi2'])
        assert result['parameter_relative_error']<.03
        if index in [0,noise_trials-1]:assert abs(model.friction-baseline.Model().fit(sample).friction)<1e-8
    report['noise']={'parameter_min':min(parameters),'parameter_max':max(parameters),'parameter_relative_error_max':float(np.max(abs(np.array(parameters)/1.1-1))),'chi2_max':max(chi2),'chi2_pass_fraction':float(np.mean(np.array(chi2)<1.5))}
    assert report['noise']['chi2_pass_fraction']>=.99
    report['hidden_parameter_extrema_check']={}
    for label,m in [('oracle',oracle),('shortcut',baseline)]:
        errors={k:[] for k in hidden}
        for gamma in [min(parameters),max(parameters)]:
            model=m.Model();model.friction=gamma
            for k,es in hidden.items():errors[k].append(score(model.predict(es),truth[k]))
        report['hidden_parameter_extrema_check'][label]={k:{'min':min(v),'max':max(v)} for k,v in errors.items()}
        if label=='oracle':assert max(max(v) for v in errors.values())<.025
        else:
            assert max(errors['single_mode_anchors'])<.025
            assert min(min(v) for k,v in errors.items() if k!='single_mode_anchors')>.025
    report['independent_checks']=independent_checks(ref,oracle,baseline)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true')
    parser.add_argument('--noise-trials',type=int,default=256)
    parser.add_argument('--output',type=Path,default=ROOT/'jobs/heated-filament-validation/summary.json')
    args=parser.parse_args()
    if args.noise_trials<1:parser.error('noise-trials must be positive')
    start=time.monotonic();report=validate(args.generate,args.noise_trials);report['seconds']=time.monotonic()-start
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'heated-filament':report},indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
