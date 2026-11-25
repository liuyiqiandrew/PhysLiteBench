from pathlib import Path
import argparse,hashlib,importlib.util,json,time
import numpy as np
S=Path('/Users/yp9846/Desktop/PhysLiteBench/staging/overnight-20261011');T=S/'tasks/magnetic-tracer';A=S/'predecessors/magnetic-tracer-r6'
def load(n,p):
    spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def err(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-14))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args();tick=time.monotonic()
    good=load('good',T/'solution/model.py');bad=load('bad',S/'scripts/magnetic_tracer_baseline.py');ref=load('ref',T/'tests/reference.py');old=load('old',A/'task/solution/model.py')
    prior=json.loads((A/'validation.json').read_text());assert all(sha(A/'task'/p)==h for p,h in prior['source_sha256'].items())
    assert (T/'tests/test_hidden.py').read_bytes()==(A/'task/tests/test_hidden.py').read_bytes()
    records=json.loads((T/'environment/data/calibration.json').read_text());assert records==json.loads((T/'tests/data/calibration.json').read_text()) and sha(T/'environment/data/calibration.json')==sha(A/'task/environment/data/calibration.json')
    inputs=[r['input'] for r in records];values=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records]);hidden=ref.hidden_inputs();truth={}
    for name,es in hidden.items():
        truth[name]=ref.predict(es);print('Independent group',name,round(time.monotonic()-tick,1),flush=True)
    frozen_path=T/'tests/data/hidden_truth.json'
    if args.freeze:frozen_path.write_text(json.dumps(dict(reference_sha256=sha(T/'tests/reference.py'),description='Full finite-mass joint valve/Kramers generator tilted by the mechanical circulation velocity, extrapolated from .0025,.00125,.000625; density and harmonic anchors retain R6 reference.',groups={n:dict(inputs=hidden[n],truth=v.tolist()) for n,v in truth.items()}),indent=2)+'\n')
    frozen=json.loads(frozen_path.read_text());assert frozen['reference_sha256']==sha(T/'tests/reference.py') and all(err(truth[n],np.array(frozen['groups'][n]['truth']))<1e-12 for n in hidden)
    distinguishing=['circulation_mean','circulation_third']
    def scores(m,g):return {n:err(m.predict_at(es,g),truth[n]) for n,es in hidden.items()}
    controls={}
    for name,m in [('oracle',good),('shortcut',bad)]:
        fit=m.Model().fit(records);res=(fit.predict(inputs)-values)/sigma;start=time.monotonic();row=dict(parameter=fit.friction,calibration_chi2=float(res@res)/(len(records)-1),hidden=scores(m,fit.friction),seconds=time.monotonic()-start)
        assert abs(fit.friction/.8-1)<.03 and row['calibration_chi2']<1.5 and row['seconds']<45
        if name=='oracle':assert max(row['hidden'].values())<.025,row
        else:assert all(row['hidden'][n]>.025 for n in distinguishing),row
        controls[name]=row
    print('CONTROLS',json.dumps(controls),flush=True)
    rng=np.random.default_rng(202610117);domain=[]
    for i in range(16):
        g=float(rng.uniform(.5,1.5));z=float(rng.uniform(.4,1.3));kap=float(rng.uniform(.1,3));temp=float(rng.uniform(.5,1.2));tb=float(rng.uniform(.4,1.4));b0=float(rng.uniform(-1,1));b1=float(rng.uniform(0,1));phase=float(rng.uniform(-np.pi,np.pi));height=float(rng.uniform(0,1.2));rate=float(rng.uniform(.1,3));d=float(rng.uniform(-.7,.7));delta=float(rng.uniform(-.6,.6));k=int(rng.integers(1,4));q=['cos','sin'][i%2]
        if i<8:
            g=(.5,1.5)[i&1];z=(.4,1.3)[(i>>1)&1];kap=(.1,3.)[(i>>2)&1];temp=(.5,1.2)[i&1];tb=(.4,1.4)[(i>>1)&1];b0=(-1.,1.)[(i>>2)&1];b1=1.;height=1.2;phase=.8;rate=(.1,3.)[i&1];d=.7;delta=(-.6,.6)[(i>>2)&1];k=[1,3][i&1]
        settings=(g,z,kap,temp,tb,b0,b1,phase,height,rate,d,delta)
        a=good.circulation(*settings,harmonic=k,quadrature=q);fine=good.circulation(*settings,n=129,harmonic=k,quadrature=q)
        es=[ref.circulation_experiment(obs,rate=rate,drag=d,contrast=delta,field=b0,modulation=b1,phase=phase,height=height,T=temp,Tb=tb,zeta=z,kappa=kap,harmonic=k,quadrature=q) for obs in ['circulation_mean','circulation_noise','circulation_third']]
        independent=ref.predict(es,g)
        field_reverse=good.circulation(g,z,kap,temp,tb,-b0,-b1,phase,height,rate,d,delta,harmonic=k,quadrature=q)*np.array([-1,1,-1])
        spatial_reverse=good.circulation(g,z,kap,temp,tb,b0,b1,-phase,height,rate,d,delta,harmonic=k,quadrature=q)*(np.array([-1,1,-1]) if q=='cos' else 1)
        relabel=good.circulation(g,z,kap,temp,tb,b0,b1,phase,height,rate,-d,-delta,harmonic=k,quadrature=q)
        row=dict(settings=settings,harmonic=k,quadrature=q,oracle=a.tolist(),independent=independent.tolist(),independent_absolute_error=float(max(abs(a-independent))),spatial_error=float(max(abs(a-fine))),field_reversal_error=float(max(abs(a-field_reverse))),spatial_reflection_error=float(max(abs(a-spatial_reverse))),valve_relabelling_error=float(max(abs(a-relabel))))
        assert row['independent_absolute_error']<3e-5,row
        assert max(row[k] for k in ['spatial_error','field_reversal_error','spatial_reflection_error','valve_relabelling_error'])<1e-8 and a[1]>0,row
        if i in [0,7,15]:
            refined=ref.predict(es,g,mass_scale=.00125,degree=6,n=33);row['joint_refinement_error']=float(max(abs(a-refined)));assert row['joint_refinement_error']<1e-5,row
        # R6 cell covariance, energy balance and density are inherited without a numerical change.
        M,D,c,C,K=good.valve_cell(g,z,kap,temp,tb,b0,rate,d,delta);previous=old.valve_cell(g,z,kap,temp,tb,b0,rate,d,delta)
        assert all(np.max(abs(np.array(a)-np.array(b)))<1e-14 for a,b in zip([M,D,c,C],previous))
        assert np.max(abs((K+K.T)/2-D))<1e-14 and abs(D[0,1])<1e-10
        assert max(abs(good.valve_moments(*settings)-old.valve_moments(*settings)))<1e-14
        domain.append(row)
        if (i+1)%4==0:print('Domain',i+1,'of16',round(time.monotonic()-tick,1),flush=True)
    limits=[]
    for temp,gamma,k,q in [(.5,.5,1,'cos'),(1.2,1.5,3,'sin')]:
        settings=(gamma,.8,.9,temp,temp,.7,.9,.8,.8,.1,.7,0.)
        a=good.circulation(*settings,harmonic=k,quadrature=q);assert abs(a[0])<1e-10 and a[1]>0
        es=[ref.circulation_experiment(obs,T=temp,Tb=temp,zeta=.8,kappa=.9,field=.7,modulation=.9,phase=.8,height=.8,rate=.1,drag=.7,contrast=0.,harmonic=k,quadrature=q) for obs in ['circulation_mean','circulation_noise','circulation_third']]
        independent=ref.predict(es,gamma);difference=float(max(abs(a-independent)));assert difference<1e-5
        limits.append(dict(kind='equal_temperature_zero_mean',oracle=a.tolist(),independent=independent.tolist(),error=difference))
    nofield=good.circulation(B0=0.,B1=0.);assert max(abs(nofield[[0,2]]))<1e-12
    limits.append(dict(kind='zero_field_odd_cumulants',oracle=nofield.tolist()))
    clean=good.predict_at(inputs,.8);rng=np.random.default_rng(905474);fits=[]
    for _ in range(256):
        noisy=clean+rng.normal(size=len(records))*sigma;sample=[dict(r,value=float(v)) for r,v in zip(records,noisy)];gf=good.Model().fit(sample);bf=bad.Model().fit(sample);res=(gf.predict(inputs)-noisy)/sigma;chi=float(res@res)/(len(records)-1)
        assert abs(gf.friction/.8-1)<.03 and abs(gf.friction-bf.friction)<1e-12 and chi<1.5;fits.append(dict(parameter=gf.friction,chi2=chi))
    parameters=np.array([r['parameter'] for r in fits]);sensitivity=[]
    for g in np.quantile(parameters,np.linspace(0,1,9)):
        a=scores(good,g);b=scores(bad,g);assert max(a.values())<.025 and all(b[n]>.025 for n in distinguishing);sensitivity.append(dict(parameter=float(g),oracle=a,shortcut=b))
    report=dict(revision=7,calibration_regenerated=False,inherited_r6_validation_sha256=sha(A/'validation.json'),controls=controls,physics=dict(domain=domain,limits=limits),noise=dict(seed=905474,trials=256,method='256 actual calibration fits and full hidden predictions at nine fitted-parameter quantiles including extrema.',fit_min=float(parameters.min()),fit_max=float(parameters.max()),max_chi2=max(r['chi2'] for r in fits),hidden_sensitivity=sensitivity),seconds=time.monotonic()-tick,source_sha256={str(p.relative_to(T)):sha(p) for p in T.rglob('*') if p.is_file() and not any(k in p.parts for k in ['__pycache__','.pytest_cache'])})
    (S/'results/magnetic-tracer-r7-validation.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',round(report['seconds'],1),flush=True)
if __name__=='__main__':main()
