from pathlib import Path
import argparse,hashlib,importlib.util,json,time
import numpy as np
from scipy.special import ive
from scipy.linalg import solve_continuous_lyapunov
S=Path('/Users/yp9846/Desktop/PhysLiteBench/staging/overnight-20261011');T=S/'tasks/magnetic-tracer'
def load(n,p):
    spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def err(a,b):return float(np.linalg.norm(a-b)/np.linalg.norm(b))
def main():
    pa=argparse.ArgumentParser();pa.add_argument('--freeze',action='store_true');args=pa.parse_args();tick=time.monotonic()
    good=load('good',T/'solution/model.py');bad=load('bad',S/'scripts/magnetic_tracer_baseline.py');ref=load('ref',T/'tests/reference.py')
    records=json.loads((T/'environment/data/calibration.json').read_text());assert records==json.loads((T/'tests/data/calibration.json').read_text())
    assert sha(T/'environment/data/calibration.json')==sha(S/'predecessors/magnetic-tracer-r4/task/environment/data/calibration.json')
    inputs=[r['input'] for r in records];y=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records]);hidden=ref.hidden_inputs();truth={}
    for name,es in hidden.items():truth[name]=ref.predict(es);print('Independent group',name,round(time.monotonic()-tick,1),flush=True)
    frozen_path=T/'tests/data/hidden_truth.json'
    if args.freeze:frozen_path.write_text(json.dumps(dict(reference_sha256=sha(T/'tests/reference.py'),description='Finite-mass four-coordinate Hermite and spatial Fourier Kramers equation, extrapolated from masses .01, .005 and .0025; harmonic anchors use full six-dimensional covariance.',groups={n:dict(inputs=hidden[n],truth=v.tolist()) for n,v in truth.items()}),indent=2)+'\n')
    frozen=json.loads(frozen_path.read_text());assert frozen['reference_sha256']==sha(T/'tests/reference.py')
    assert all(err(truth[n],np.array(frozen['groups'][n]['truth']))<1e-12 for n in hidden)
    def scores(m,g):return {n:err(m.predict_at(es,g),truth[n]) for n,es in hidden.items()}
    controls={}
    for name,m in [('oracle',good),('shortcut',bad)]:
        fit=m.Model().fit(records);res=(fit.predict(inputs)-y)/sigma;start=time.monotonic();row=dict(parameter=fit.friction,calibration_chi2=float(res@res)/(len(records)-1),hidden=scores(m,fit.friction),seconds=time.monotonic()-start)
        assert abs(fit.friction/.8-1)<.03 and row['calibration_chi2']<1.5 and row['seconds']<45
        if name=='oracle':assert max(row['hidden'].values())<.025,row
        else:assert all(row['hidden'][n]>.025 for n in ['hot_contact_density','cold_contact_density','zero_potential_density']),row
        controls[name]=row
    print('CONTROLS',json.dumps(controls),flush=True)
    rng=np.random.default_rng(202610115);domain=[]
    for i in range(48):
        gamma=float(rng.uniform(.5,1.5));zeta=float(rng.uniform(.4,1.3));kappa=float(rng.uniform(.1,3));temp=float(rng.uniform(.5,1.2));tb=float(rng.uniform(.4,1.4));b0=float(rng.uniform(-1,1));b1=float(rng.uniform(0,1));phase=float(rng.uniform(-np.pi,np.pi));height=float(rng.uniform(0,1.2))
        if i<16:
            gamma=(.5,1.5)[i&1];zeta=(.4,1.3)[(i>>1)&1];kappa=(.1,3.)[(i>>2)&1];temp=(.5,1.2)[i&1];tb=(.4,1.4)[(i>>3)&1];b0=(-1.,1.)[(i>>2)&1];b1=1.;height=1.2;phase=.8
        settings=(gamma,zeta,kappa,temp,tb,b0,b1,phase,height)
        x,p,drift,diff=good.channel_density(*settings);xx,pp,_,_=good.channel_density(*settings,n=129)
        value=good.channel_moments(*settings);fine=np.array([v for k in [1,2,3] for v in [np.mean(pp*np.cos(k*xx)),np.mean(pp*np.sin(k*xx))]])
        es=[ref.channel_experiment(b0,b1,phase,height,temp,tb,zeta,kappa,h,q) for h in [1,2,3] for q in ['cos','sin']]
        independent=ref.predict(es,gamma)
        reflected=good.channel_moments(gamma,zeta,kappa,temp,tb,b0,b1,-phase,height)*np.array([1,-1]*3)
        field_reversed=good.channel_moments(gamma,zeta,kappa,temp,tb,-b0,-b1,phase,height)
        row=dict(settings=settings,independent_error=float(max(abs(value-independent))),spatial_error=float(max(abs(value-fine))),reflection_error=float(max(abs(value-reflected))),field_reversal_error=float(max(abs(value-field_reversed))),minimum_density=float(p.min()),normalization_error=float(abs(p.mean()-1)),stationary_winding_current=float(np.mean(p*drift[:,0])),stationary_transverse_current=float(np.mean(p*drift[:,1])))
        assert row['independent_error']<2e-5,row
        assert max(row[k] for k in ['spatial_error','reflection_error','field_reversal_error','normalization_error'])<1e-8,row
        assert max(abs(row[k]) for k in ['stationary_winding_current','stationary_transverse_current'])<1e-9,row
        assert row['minimum_density']>0
        if i in [0,3,7,12,19,27,35,47]:
            higher=ref.predict(es,gamma,mass_scale=.005,degree=8,n=41);row['joint_refinement_error']=float(max(abs(value-higher)));assert row['joint_refinement_error']<5e-6,row
        # The frozen OU bath variables exchange energy conservatively through
        # their reciprocal contact coupling and the antisymmetric Lorentz term.
        A=np.block([[-gamma*np.eye(2)+b0*good.ROTATION,np.sqrt(zeta/kappa)*np.eye(2)],[-np.sqrt(zeta/kappa)*np.eye(2),-np.eye(2)/kappa]])
        Q=np.diag([2*gamma*temp]*2+[2*tb/kappa]*2);cov=solve_continuous_lyapunov(A,-Q)
        row['fast_balance_error']=float(np.max(abs(A@cov+cov@A.T+Q)));row['fast_energy_error']=float(abs(gamma*(2*temp-np.trace(cov[:2,:2]))+(2*tb-np.trace(cov[2:,2:]))/kappa));row['fast_entropy']=float(gamma*(np.trace(cov[:2,:2])-2*temp)/temp+(np.trace(cov[2:,2:])-2*tb)/(kappa*tb))
        assert row['fast_balance_error']<1e-10 and row['fast_energy_error']<1e-10 and row['fast_entropy']>-1e-10 and np.linalg.eigvalsh(cov).min()>0
        domain.append(row)
        if (i+1)%4==0:print('Domain checks',i+1,'of48',round(time.monotonic()-tick,1),flush=True)
    limits=[]
    for gamma in [.5,1.5]:
        for temp in [.5,1.2]:
            for height in [0.,1.2]:
                values=good.channel_moments(gamma,.8,.9,temp,temp,.7,.9,.8,height);boltzmann=np.array([v for k in [1,2,3] for v in [(-1)**k*ive(k,height/temp)/ive(0,height/temp),0.]])
                assert max(abs(values-boltzmann))<1e-9
                limits.append(dict(type='equal_temperature',gamma=gamma,temperature=temp,height=height,error=float(max(abs(values-boltzmann)))))
    for kappa in [.1,1.,3.]:
        values=good.channel_moments(.8,1.,kappa,.5,1.4,.7,0.,.8,.9);eq=bad.channel_moments(.8,1.,kappa,.5,1.4,.7,0.,.8,.9);assert max(abs(values-eq))<1e-9
        limits.append(dict(type='uniform_field',kappa=kappa,error=float(max(abs(values-eq)))))
    clean=good.predict_at(inputs,.8);rng=np.random.default_rng(905472);fits=[]
    for _ in range(256):
        noisy=clean+rng.normal(size=len(records))*sigma;sample=[dict(r,value=float(v)) for r,v in zip(records,noisy)];fit=good.Model().fit(sample);bf=bad.Model().fit(sample);res=(fit.predict(inputs)-noisy)/sigma;chi=float(res@res)/(len(records)-1)
        assert abs(fit.friction/.8-1)<.03 and abs(fit.friction-bf.friction)<1e-12 and chi<1.5;fits.append(dict(parameter=fit.friction,calibration_chi2=chi))
    params=np.array([r['parameter'] for r in fits]);sensitivity=[]
    for g in np.quantile(params,np.linspace(0,1,9)):
        a=scores(good,g);b=scores(bad,g);assert max(a.values())<.025 and all(b[n]>.025 for n in ['hot_contact_density','cold_contact_density','zero_potential_density']);sensitivity.append(dict(parameter=float(g),oracle=a,shortcut=b))
    report=dict(revision=5,calibration_regenerated=False,controls=controls,physics=dict(domain=domain,limits=limits),noise=dict(seed=905472,trials=256,method='256 actual calibration fits; all hidden predictions evaluated at nine fitted-parameter quantiles including both extrema, not 256 full hidden predictions.',fit_min=float(params.min()),fit_max=float(params.max()),max_chi2=max(r['calibration_chi2'] for r in fits),hidden_sensitivity=sensitivity),seconds=time.monotonic()-tick,source_sha256={str(p.relative_to(T)):sha(p) for p in T.rglob('*') if p.is_file() and not any(k in p.parts for k in ['__pycache__','.pytest_cache'])})
    (S/'results/magnetic-tracer-r5-validation.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',round(report['seconds'],1),flush=True)
if __name__=='__main__':main()
