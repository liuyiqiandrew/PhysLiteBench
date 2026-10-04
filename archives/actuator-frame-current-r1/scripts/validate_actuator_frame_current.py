"""Calibration, kinetic-cell, stationary-current and local harness checks."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/actuator-frame-current'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


good=load(TASK/'solution/model.py','oracle')
bad=load(ROOT/'scripts/actuator_frame_current_baseline.py','shortcut')
ref=load(TASK/'tests/reference.py','reference')


def validate(count):
    records=json.loads((TASK/'tests/data/calibration.json').read_text())
    inputs=[r['input'] for r in records];sigma=np.array([r['sigma'] for r in records]);clean=ref.predict(inputs,.8)
    hidden=ref.hidden_inputs();truth={key:ref.predict(es,.8) for key,es in hidden.items()}
    def scores(module,strength):
        return {key:float(np.sqrt(np.mean((module.predict_at(es,strength)-truth[key])**2)/np.mean(truth[key]**2))) for key,es in hidden.items()}
    controls={}
    for label,module in [('oracle',good),('shortcut',bad)]:
        model=module.Model().fit(records);res=(model.predict(inputs)-np.array([r['value'] for r in records]))/sigma
        controls[label]=dict(noise_strength=model.noise_strength,calibration_chi2=float(res@res/(len(records)-1)),hidden=scores(module,model.noise_strength))
    rng=np.random.default_rng(927133);samples=[]
    for _ in range(count):
        observed=clean+sigma*rng.normal(size=len(inputs))
        noisy=[dict(input=e,value=float(v),sigma=float(s)) for e,v,s in zip(inputs,observed,sigma)]
        model=good.Model().fit(noisy);other=bad.Model().fit(noisy)
        assert abs(model.noise_strength-other.noise_strength)<1e-10
        res=(model.predict(inputs)-observed)/sigma
        samples.append(dict(noise_strength=model.noise_strength,chi2=float(res@res/(len(inputs)-1)),oracle=scores(good,model.noise_strength),shortcut=scores(bad,model.noise_strength)))
    all_inputs=inputs+sum(hidden.values(),[])
    corners=[ref.experiment(a,c,p,k,b,f,o) for a in [.25,4.] for c in [0.,.65] for p in [0.,1.5] for k,b,f in [(2,1.,.4),(-2,-1.,-.4),(0,.8,0.)] for o in ['cosine','current']]
    reference_error=corner_error=refinement=quadrature=mirror=uniform=cell_error=cell_diffusion_error=minimum_eigenvalue=0.
    minimum_eigenvalue=1e9
    for strength in [.55,.8,1.1]:
        reference_error=max(reference_error,float(np.max(abs(good.predict_at(all_inputs,strength)-ref.predict(all_inputs,strength)))))
        corner_error=max(corner_error,float(np.max(abs(good.predict_at(corners,strength)-ref.predict(corners,strength)))))
        refinement=max(refinement,float(np.max(abs(ref.predict(corners,strength,128)-ref.predict(corners,strength,256)))))
        for e in corners+sum(hidden.values(),[]):
            value=good.response(e,strength)
            quadrature=max(quadrature,abs(value-good.response(e,strength,1024)))
            reverse=dict(e,winding=-e['winding'],twist=-e['twist'],push=-e['push'])
            mirror=max(mirror,abs(good.response(reverse,strength)-(-value if e['observable']=='current' else value)))
            if e['contrast']==0 and e['potential']==0:
                exact=0. if e['observable']=='cosine' else e['push']+e['ratio']/(1+e['ratio'])*strength*e['winding']
                uniform=max(uniform,abs(value-exact))
            x,drift,diffusion,_=ref.cell_coefficients(e['ratio'],e['contrast'],e['winding'],e['twist'],64,0.)
            a=1+e['contrast']*np.cos(x);ap=-e['contrast']*np.sin(x);theta_prime=e['winding']+e['twist']*np.cos(x);c=e['ratio']/(1+e['ratio'])
            exact=c*np.array([a*ap,a*a*theta_prime]).T
            cell_error=max(cell_error,float(np.max(abs(drift-exact))))
            cell_diffusion_error=max(cell_diffusion_error,float(np.max(abs(diffusion-a[:,None,None]**2*np.eye(2)))))
            minimum_eigenvalue=min(minimum_eigenvalue,float(np.linalg.eigvalsh(diffusion).min()))
    equivalence=max(float(np.max(abs(good.predict_at(inputs,s)-bad.predict_at(inputs,s)))) for s in [.55,.8,1.1])
    noiseless=[]
    for strength in [.55,.65,.8,.95,1.1]:
        values=ref.predict(inputs,strength)
        model=good.Model().fit([dict(input=e,value=float(v),sigma=.0006) for e,v in zip(inputs,values)])
        noiseless.append(dict(true=strength,fitted=model.noise_strength))
    profile=[]
    for strength in np.linspace(.55,1.1,111):
        res=(good.predict_at(inputs,strength)-clean)/sigma
        profile.append([float(strength),float(res@res)])
    minima=[i for i in range(1,len(profile)-1) if profile[i][1]<profile[i-1][1] and profile[i][1]<profile[i+1][1]]
    monotonic=np.diff(np.array([good.predict_at(inputs,s) for s in np.linspace(.55,1.1,31)]),axis=0)
    # Conservative stationary x balance checked spectrally at hard corners.
    stationary_residual=0.;min_density=1.
    for ratio in [.25,4.]:
        for contrast in [0.,.65]:
            for strength in [.55,1.1]:
                x=2*np.pi*np.arange(512)/512;a=1+contrast*np.cos(x);ap=-contrast*np.sin(x);c=ratio/(1+ratio)
                logp=(c-2)*np.log(a)-1.5*np.cos(x)/(strength*a)
                density=np.exp(logp-logp.max());density/=density.sum()*2*np.pi/512
                wave=np.fft.fftfreq(512,d=1/512)
                flux=(1.5*np.sin(x)+c*strength*a*ap)*density-np.fft.ifft(1j*wave*np.fft.fft(strength*a*a*density)).real
                stationary_residual=max(stationary_residual,float(max(abs(flux))))
                min_density=min(min_density,float(density.min()))
    report=dict(task='actuator-frame-current',revision=1,noise_trials=count,controls=controls,noise_samples=samples,checks=dict(calibration_equivalence=equivalence,reference_error=reference_error,corner_reference_error=corner_error,reference128_to256=refinement,oracle512_to1024=quadrature,frame_reversal=mirror,uniform_exact_current=uniform,cell_drift_error=cell_error,cell_diffusion_error=cell_diffusion_error,minimum_diffusion_eigenvalue=minimum_eigenvalue,stationary_x_flux=stationary_residual,minimum_density=min_density,minimum_calibration_monotonic_increment=float(monotonic.min()),noiseless_fits=noiseless,loss_profile=profile,loss_profile_minima=minima),finite_mass_prototype='results/actuator-frame-current-prototype.json')
    report['summary']=dict(calibration_passes=sum(s['chi2']<1.5 for s in samples),parameter_passes=sum(abs(s['noise_strength']/.8-1)<.03 for s in samples),oracle_passes=sum(max(s['oracle'].values())<.04 for s in samples),shortcut_prediction_passes=sum(max(s['shortcut'].values())<.04 for s in samples),oracle_worst=max(max(s['oracle'].values()) for s in samples),shortcut_current_best=min(s['shortcut'][key] for s in samples for key in hidden if key!='position_anchors'),shortcut_position_worst=max(s['shortcut']['position_anchors'] for s in samples),max_chi2=max(s['chi2'] for s in samples))
    assert report['summary']['calibration_passes']==count
    assert report['summary']['parameter_passes']==count
    assert report['summary']['oracle_passes']==count
    assert report['summary']['shortcut_prediction_passes']==0
    assert report['summary']['shortcut_position_worst']<.04
    assert reference_error<1e-6 and corner_error<1e-6 and quadrature<1e-10
    assert equivalence<1e-12 and mirror<1e-12 and uniform<1e-12
    assert stationary_residual<1e-10 and min_density>0 and minimum_eigenvalue>0
    assert cell_error<1e-12 and cell_diffusion_error<1e-12
    assert monotonic.min()>0 and len(minima)==1
    assert max(abs(s['fitted']-s['true']) for s in noiseless)<1e-6
    return report


def local_checks():
    results={}
    for label,source in [('oracle',TASK/'solution/model.py'),('shortcut',ROOT/'scripts/actuator_frame_current_baseline.py')]:
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);shutil.copy2(source,p/'model.py');shutil.copy2(TASK/'environment/test_public.py',p/'test_public.py');shutil.copytree(TASK/'environment/data',p/'data');shutil.copytree(TASK/'tests',p/'private')
            env=dict(os.environ,PYTHONPATH=str(p),OPENBLAS_NUM_THREADS='1')
            run=subprocess.run([sys.executable,'-m','pytest','-q','test_public.py','private/test_hidden.py'],cwd=p,env=env,text=True,capture_output=True)
            results[label]=dict(returncode=run.returncode,stdout=run.stdout,stderr=run.stderr)
            assert (run.returncode==0)==(label=='oracle')
            if label=='shortcut':assert '3 failed, 5 passed' in run.stdout
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');parser.add_argument('--count',type=int,default=256);args=parser.parse_args()
    if args.generate:ref.generate_data()
    result=validate(args.count);result['local_controls']=local_checks()
    output=ROOT/'results/actuator-frame-current-validation.json';output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'summary':result['summary'],'checks':{k:v for k,v in result['checks'].items() if k not in ['loss_profile']},'local_controls':result['local_controls']},indent=2))
