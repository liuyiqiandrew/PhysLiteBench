from pathlib import Path
import argparse,hashlib,importlib.util,json,time
import numpy as np
S=Path('/Users/yp9846/Desktop/PhysLiteBench/staging/overnight-20261011');T=S/'tasks/rotating-reservoir';PREV=S/'predecessors/rotating-reservoir-r10'
def load(n,p):
    spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def err(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-14))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args();tick=time.monotonic()
    good=load('good',T/'solution/model.py');bad=load('bad',S/'scripts/rotating_reservoir_baseline.py');ref=load('ref',T/'tests/reference.py');old=load('old',PREV/'task/solution/model.py')
    previous=json.loads((PREV/'validation.json').read_text());assert all(sha(PREV/'task'/p)==h for p,h in previous['source_sha256'].items())
    assert (PREV/'task/solution/model.py').read_text().split('def predict_at(')[0] in (T/'solution/model.py').read_text()
    assert (T/'tests/test_hidden.py').read_bytes()==(PREV/'task/tests/test_hidden.py').read_bytes()
    assert sha(T/'environment/data/calibration.json')==sha(PREV/'task/environment/data/calibration.json')==sha(T/'tests/data/calibration.json')
    records=json.loads((T/'environment/data/calibration.json').read_text());inputs=[r['input'] for r in records];values=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records]);hidden=ref.hidden_inputs();truth={}
    for name,es in hidden.items():
        truth[name]=ref.predict(es);print('Independent group',name,round(time.monotonic()-tick,1),flush=True)
    path=T/'tests/data/hidden_truth.json'
    if args.freeze:path.write_text(json.dumps(dict(reference_sha256=sha(T/'tests/reference.py'),description='Positive OU/swap velocity-space generator with explicit bath heat marks and angular Fourier transport; 81/162 grid extrapolation.',groups={n:dict(inputs=hidden[n],truth=v.tolist()) for n,v in truth.items()}),indent=2)+'\n')
    frozen=json.loads(path.read_text());assert frozen['reference_sha256']==sha(T/'tests/reference.py') and all(err(truth[n],np.array(frozen['groups'][n]['truth']))<1e-12 for n in hidden)
    distinguishing=['angle_gated_hot_rotor_third','angle_gated_cold_rotor_third','angle_gated_noise']
    def scores(m,g):return {name:err(m.predict_at(es,g),truth[name]) for name,es in hidden.items()}
    controls={}
    for name,m in [('oracle',good),('shortcut',bad)]:
        start=time.monotonic();fit=m.Model().fit(records);res=(fit.predict(inputs)-values)/sigma;row=dict(parameter=fit.drag,calibration_chi2=float(res@res)/(len(records)-1),hidden=scores(m,fit.drag),seconds=time.monotonic()-start)
        assert abs(fit.drag/.67-1)<.03 and row['calibration_chi2']<1.5 and row['seconds']<45
        if name=='oracle':assert max(row['hidden'].values())<.04,row
        else:assert all(row['hidden'][n]>.04 for n in distinguishing),row
        controls[name]=row
    print('CONTROLS',json.dumps(controls),flush=True)
    rng=np.random.default_rng(202610117);domain=[]
    for i in range(16):
        ta=float(rng.uniform(.4,2.));tb=float(rng.uniform(.4,1.4));lam=float(rng.uniform(.2,3));eta=float(rng.uniform(0,1));c=float(rng.uniform(0,.8));k=int(rng.integers(1,4));drag=float(rng.uniform(.4,1.1))
        if i<8:
            ta=(.4,2.)[i&1];tb=(.4,1.4)[(i>>1)&1];lam=(.2,3.)[(i>>2)&1];eta=(0.,1.)[i&1];c=.8;k=(1,3)[(i>>1)&1];drag=(.4,1.1)[i&1]
        e=ref.gated_experiment(ta,tb,lam,eta,c,k);a=good.angular_hermite(e,drag);b=good.angular_hermite(e,drag,degree=40)
        fields=['mean','noise','third_cumulant'];es=[dict(e,statistic=f) for f in fields];independent=ref.predict(es,drag,points=61);va=np.array([a[f] for f in fields]);vb=np.array([b[f] for f in fields])
        # A mixed norm remains defined at equilibrium, where odd heat cumulants vanish.
        row=dict(input=e,drag=drag,oracle=a,independent=independent.tolist(),independent_error=err(va,independent),degree_error=err(va,vb))
        assert row['independent_error']<.015 and row['degree_error']<.005,row
        assert a['noise']>0
        if i in [0,3,7,15]:
            refined=ref.predict(es,drag,points=101);row['refined_independent_error']=err(va,refined);assert row['refined_independent_error']<.01,row
        zero=good.angular_hermite(dict(e,detector_contrast=0.),drag)
        ungated=old.kinetic_statistics(e,drag)
        row['zero_contrast_error']=err(np.array([zero[f] for f in fields]),np.array([ungated[f] for f in fields]));assert row['zero_contrast_error']<.001,row
        reverse=good.angular_hermite(dict(e,detector_harmonic=-k),drag)
        row['angle_reflection_error']=err(va,np.array([reverse[f] for f in fields]));assert row['angle_reflection_error']<1e-10
        negative=good.angular_hermite(dict(e,detector_contrast=-c),drag)
        row['gain_phase_shift_error']=err(va,np.array([negative[f] for f in fields]));assert row['gain_phase_shift_error']<1e-10
        row['mean_unchanged_error']=float(abs(a['mean']-zero['mean']));assert row['mean_unchanged_error']<1e-10
        if i in [0,7,15]:
            rapid=good.angular_hermite(e,drag,phase_mixing_rate=1e6);averaged=bad.angular_hermite(e,drag)
            row['rapid_phase_error']=err(np.array([rapid[f] for f in fields]),np.array([averaged[f] for f in fields]));assert row['rapid_phase_error']<1e-5,row
        domain.append(row)
        if (i+1)%4==0:print('Domain',i+1,'of16',round(time.monotonic()-tick,1),flush=True)
    equilibrium=[]
    for temp,eta,k in [(.4,0.,1),(1.,1.,2),(1.4,.5,3)]:
        e=ref.gated_experiment(temp,temp,1.3,eta,.8,k);a=good.angular_hermite(e,.67);assert abs(a['mean'])<1e-9 and abs(a['third_cumulant'])<1e-8 and a['noise']>0,a
        equilibrium.append(dict(input=e,result=a))
    clean=good.predict_at(inputs,.67);rng=np.random.default_rng(146835);fits=[]
    for _ in range(256):
        noisy=clean+rng.normal(size=len(records))*sigma;sample=[dict(r,value=float(v)) for r,v in zip(records,noisy)];gf=good.Model().fit(sample);bf=bad.Model().fit(sample);res=(gf.predict(inputs)-noisy)/sigma;chi=float(res@res)/(len(records)-1)
        assert abs(gf.drag/.67-1)<.03 and abs(gf.drag-bf.drag)<1e-12 and chi<1.5;fits.append(dict(parameter=gf.drag,chi2=chi))
    params=np.array([r['parameter'] for r in fits]);sensitivity=[]
    for g in np.quantile(params,np.linspace(0,1,9)):
        a=scores(good,g);b=scores(bad,g);assert max(a.values())<.04 and all(b[n]>.04 for n in distinguishing);sensitivity.append(dict(parameter=float(g),oracle=a,shortcut=b));print('Noise sensitivity',len(sensitivity),'of9',round(time.monotonic()-tick,1),flush=True)
    report=dict(revision=11,calibration_regenerated=False,inherited_r10_validation_sha256=sha(PREV/'validation.json'),controls=controls,physics=dict(domain=domain,equilibrium=equilibrium),noise=dict(seed=146835,trials=256,method='256 actual calibration fits and full hidden predictions at nine fitted-parameter quantiles including extrema.',fit_min=float(params.min()),fit_max=float(params.max()),max_chi2=max(r['chi2'] for r in fits),hidden_sensitivity=sensitivity),seconds=time.monotonic()-tick,source_sha256={str(p.relative_to(T)):sha(p) for p in T.rglob('*') if p.is_file() and not any(k in p.parts for k in ['__pycache__','.pytest_cache'])})
    (S/'results/rotating-reservoir-r11-validation.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',round(report['seconds'],1),flush=True)
if __name__=='__main__':main()
