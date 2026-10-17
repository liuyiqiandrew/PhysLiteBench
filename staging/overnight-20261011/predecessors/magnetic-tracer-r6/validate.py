from pathlib import Path
import argparse,hashlib,importlib.util,json,time
import numpy as np
from scipy.special import ive
S=Path('/Users/yp9846/Desktop/PhysLiteBench/staging/overnight-20261011');T=S/'tasks/magnetic-tracer'
def load(n,p):
    spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def err(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-14))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args();tick=time.monotonic()
    good=load('good',T/'solution/model.py');bad=load('bad',S/'scripts/magnetic_tracer_baseline.py');ref=load('ref',T/'tests/reference.py')
    prev=S/'predecessors/magnetic-tracer-r5';old=load('old',prev/'task/solution/model.py');previous=json.loads((prev/'validation.json').read_text())
    assert all(sha(prev/'task'/p)==h for p,h in previous['source_sha256'].items())
    oldsource=(prev/'task/solution/model.py').read_text();newsource=(T/'solution/model.py').read_text()
    assert oldsource[:oldsource.index('def channel_prediction')].rstrip() in newsource
    records=json.loads((T/'environment/data/calibration.json').read_text());assert records==json.loads((T/'tests/data/calibration.json').read_text())
    assert sha(T/'environment/data/calibration.json')==sha(prev/'task/environment/data/calibration.json')
    inputs=[r['input'] for r in records];y=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records]);hidden=ref.hidden_inputs();truth={}
    for name,es in hidden.items():
        truth[name]=ref.predict(es);print('Independent group',name,round(time.monotonic()-tick,1),flush=True)
    path=T/'tests/data/hidden_truth.json'
    if args.freeze:path.write_text(json.dumps(dict(reference_sha256=sha(T/'tests/reference.py'),description='Explicit two-state finite-mass Kramers equation in Fourier/Hermite representation, masses .01,.005,.0025; prior preparations retain R5 reference.',groups={n:dict(inputs=hidden[n],truth=v.tolist()) for n,v in truth.items()}),indent=2)+'\n')
    frozen=json.loads(path.read_text());assert frozen['reference_sha256']==sha(T/'tests/reference.py')
    assert all(err(truth[n],np.array(frozen['groups'][n]['truth']))<1e-12 for n in hidden)
    distinguishing=['valve_hot_memory','valve_cold_memory','valve_zero_potential']
    def scores(m,g):return {n:err(m.predict_at(es,g),truth[n]) for n,es in hidden.items()}
    controls={}
    for name,m in [('oracle',good),('shortcut',bad)]:
        fit=m.Model().fit(records);res=(fit.predict(inputs)-y)/sigma;start=time.monotonic();row=dict(parameter=fit.friction,calibration_chi2=float(res@res)/(len(records)-1),hidden=scores(m,fit.friction),seconds=time.monotonic()-start)
        assert abs(fit.friction/.8-1)<.03 and row['calibration_chi2']<1.5 and row['seconds']<45
        if name=='oracle':assert max(row['hidden'].values())<.025,row
        else:assert all(row['hidden'][n]>.025 for n in distinguishing),row
        controls[name]=row
    print('CONTROLS',json.dumps(controls),flush=True)
    rng=np.random.default_rng(202610116);domain=[]
    for i in range(32):
        g=float(rng.uniform(.5,1.5));z=float(rng.uniform(.4,1.3));kap=float(rng.uniform(.1,3));temp=float(rng.uniform(.5,1.2));tb=float(rng.uniform(.4,1.4));b0=float(rng.uniform(-1,1));b1=float(rng.uniform(0,1));phase=float(rng.uniform(-np.pi,np.pi));height=float(rng.uniform(0,1.2));rate=float(rng.uniform(.1,3));d=float(rng.uniform(-.7,.7));delta=float(rng.uniform(-.6,.6))
        if i<8:
            g=(.5,1.5)[i&1];z=(.4,1.3)[(i>>1)&1];kap=(.1,3.)[(i>>2)&1];temp=(.5,1.2)[i&1];tb=(.4,1.4)[(i>>1)&1];b0=(-1.,1.)[(i>>2)&1];b1=1.;height=1.2;phase=.8;rate=(.1,3.)[i&1];d=.7;delta=(-.6,.6)[(i>>2)&1]
        settings=(g,z,kap,temp,tb,b0,b1,phase,height,rate,d,delta)
        value=good.valve_moments(*settings);fine=good.valve_moments(*settings,n=129)
        es=[ref.valve_experiment(rate,d,delta,field=b0,modulation=b1,phase=phase,height=height,T=temp,Tb=tb,zeta=z,kappa=kap,harmonic=h,quadrature=q) for h in [1,2,3] for q in ['cos','sin']]
        independent=ref.predict(es,g)
        reflected=good.valve_moments(g,z,kap,temp,tb,b0,b1,-phase,height,rate,d,delta)*np.array([1,-1]*3)
        reversed_field=good.valve_moments(g,z,kap,temp,tb,-b0,-b1,phase,height,rate,d,delta)
        relabelled=good.valve_moments(g,z,kap,temp,tb,b0,b1,phase,height,rate,-d,-delta)
        zero=good.valve_moments(g,z,kap,temp,tb,b0,b1,phase,height,rate,0.,0.)
        no_valve=old.channel_moments(g,z,kap,temp,tb,b0,b1,phase,height)
        averaged=bad.valve_moments(g,z,kap,temp,tb,b0,b1,phase,height,rate,d,delta)
        rapid=good.valve_moments(g,z,kap,temp,tb,b0,b1,phase,height,1e5,d,delta)
        row=dict(settings=settings,independent_error=float(max(abs(value-independent))),spatial_error=float(max(abs(value-fine))),reflection_error=float(max(abs(value-reflected))),field_reversal_error=float(max(abs(value-reversed_field))),label_error=float(max(abs(value-relabelled))),zero_contrast_error=float(max(abs(zero-no_valve))),rapid_switch_error=float(max(abs(rapid-averaged))))
        assert row['independent_error']<2e-5,row
        assert max(row[k] for k in ['spatial_error','reflection_error','field_reversal_error','label_error','zero_contrast_error'])<1e-8,row
        assert row['rapid_switch_error']<5e-6,row
        if i in [0,3,7,15,23,31]:
            high=ref.predict(es,g,mass_scale=.005,degree=6,n=33);row['joint_refinement_error']=float(max(abs(value-high)));assert row['joint_refinement_error']<5e-6,row
        M,D,derivative,covs=good.valve_cell(g,z,kap,temp,tb,b0,rate,d,delta)
        energy=0.;entropy=0.;balance=[]
        for j,sign in enumerate([-1,1]):
            drag=g*(1+sign*d);ts=temp*(1+sign*delta)
            A=np.block([[-drag*np.eye(2)+b0*good.ROTATION,np.sqrt(z/kap)*np.eye(2)],[-np.sqrt(z/kap)*np.eye(2),-np.eye(2)/kap]])
            Q=np.diag([2*drag*ts]*2+[2*tb/kap]*2)
            balance.append(float(np.max(abs(A@covs[j]+covs[j]@A.T+rate*(covs[1-j]-covs[j])+.5*Q))))
            heat=drag*(np.trace(covs[j][:2,:2])-ts);contact=(np.trace(covs[j][2:,2:])-tb)/kap
            energy+=heat+contact;entropy+=heat/ts+contact/tb
        row.update(fast_balance_error=max(balance),fast_energy_error=float(abs(energy)),fast_entropy=float(entropy),minimum_covariance_eigenvalue=float(min(np.linalg.eigvalsh(a).min() for a in covs)),minimum_diffusion_eigenvalue=float(np.linalg.eigvalsh(D).min()))
        assert max(balance)<1e-10 and abs(energy)<1e-10 and entropy>-1e-10 and row['minimum_covariance_eigenvalue']>0 and row['minimum_diffusion_eigenvalue']>0
        domain.append(row)
        if (i+1)%4==0:print('Domain checks',i+1,'of32',round(time.monotonic()-tick,1),flush=True)
    limits=[]
    for g in [.5,1.5]:
        for temp in [.5,1.2]:
            for height in [0.,1.2]:
                value=good.valve_moments(g,.8,.9,temp,temp,.7,.9,.8,height,.1,.7,0.)
                expected=np.array([v for k in [1,2,3] for v in [(-1)**k*ive(k,height/temp)/ive(0,height/temp),0.]])
                difference=float(max(abs(value-expected)));assert difference<1e-9
                limits.append(dict(type='equal_temperature',gamma=g,temperature=temp,height=height,error=difference))
    clean=good.predict_at(inputs,.8);rng=np.random.default_rng(905473);fits=[]
    for _ in range(256):
        noisy=clean+rng.normal(size=len(records))*sigma;sample=[dict(r,value=float(v)) for r,v in zip(records,noisy)];fit=good.Model().fit(sample);bf=bad.Model().fit(sample);res=(fit.predict(inputs)-noisy)/sigma;chi=float(res@res)/(len(records)-1)
        assert abs(fit.friction/.8-1)<.03 and abs(fit.friction-bf.friction)<1e-12 and chi<1.5;fits.append(dict(parameter=fit.friction,calibration_chi2=chi))
    params=np.array([r['parameter'] for r in fits]);sensitivity=[]
    for g in np.quantile(params,np.linspace(0,1,9)):
        a=scores(good,g);b=scores(bad,g);assert max(a.values())<.025 and all(b[n]>.025 for n in distinguishing);sensitivity.append(dict(parameter=float(g),oracle=a,shortcut=b))
    report=dict(revision=6,calibration_regenerated=False,inherited_r5_validation_sha256=sha(prev/'validation.json'),controls=controls,physics=dict(domain=domain,limits=limits),noise=dict(seed=905473,trials=256,method='256 actual calibration fits; hidden predictions at nine fitted-parameter quantiles including extrema.',fit_min=float(params.min()),fit_max=float(params.max()),max_chi2=max(r['calibration_chi2'] for r in fits),hidden_sensitivity=sensitivity),seconds=time.monotonic()-tick,source_sha256={str(p.relative_to(T)):sha(p) for p in T.rglob('*') if p.is_file() and not any(k in p.parts for k in ['__pycache__','.pytest_cache'])})
    (S/'results/magnetic-tracer-r6-validation.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',round(report['seconds'],1),flush=True)
if __name__=='__main__':main()
