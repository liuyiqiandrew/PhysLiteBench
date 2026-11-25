from pathlib import Path
import hashlib,importlib.util,json,time,types
import numpy as np
S=Path('/Users/yp9846/Desktop/PhysLiteBench/staging/overnight-20261011');T=S/'tasks/reaction-diffusion';A=S/'predecessors/reaction-diffusion-r8'
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def score(model,cases):
    out=[]
    for t,x,initial,truth in cases:
        y=model.predict(t,x,initial);mean=np.trapezoid(initial,x,axis=0)/.001;out.append(float(max(np.linalg.norm(y-truth,axis=(0,1))/np.linalg.norm(truth-mean,axis=(0,1)))))
    return out
def main():
    tick=time.monotonic();good=load('g',T/'solution/model.py');bad=load('b',S/'scripts/reaction_baseline.py');ref=load('r',T/'tests/reference.py');old=json.loads((A/'validation.json').read_text())
    for name in ['solution/model.py','tests/reference.py','tests/test_hidden.py','environment/README.md','environment/data/calibration.npz','tests/data/calibration.npz','task.toml','instruction.md']:
        assert sha(T/name)==sha(A/'task'/name)==old['source_sha256'][name]
    data=ref.load_data();cases=ref.hidden_cases();controls={}
    for name,m in [('oracle',good),('shortcut',bad)]:
        fit=m.TransportModel().fit(data);row=ref.metrics(fit,data);assert row['calibration_reduced_chi2']<1.5 and row['relative_diffusivity_error']<.05
        assert max(row['hidden_nrmse'])<.05 if name=='oracle' else min(row['hidden_nrmse'])>.05;controls[name]=row
    print('CONTROLS',json.dumps(controls),flush=True)
    # Independent finite-rate four-species model of the immobile-binding assumption.
    source=(T/'tests/reference.py').read_text();assert source.count('np.array([1.,4.,2.,.15])')==1
    separate=types.ModuleType('immobile_reference');separate.__file__=str(T/'tests/reference.py');exec(compile(source.replace('np.array([1.,4.,2.,.15])','np.array([1.,4.,2.,0.])'),str(T/'tests/reference.py'),'exec'),separate.__dict__)
    independent=[];m=bad.TransportModel();m.diffusivity=1e-9
    for i,(t,x,initial,_) in enumerate(cases):
        predicted=m.predict(t,x,initial);truth=separate.trajectory(t,x,initial,rate=40.,refinement=4);mean=np.trapezoid(initial,x,axis=0)/.001;error=float(np.linalg.norm(predicted-truth)/np.linalg.norm(truth-mean));assert error<.003
        independent.append(dict(case=i,nrmse=error));print('Independent immobile-binding case',i,flush=True)
    checks=[];x=np.linspace(0,.001,65);z=np.pi*x/.001;t=np.linspace(0,180,19)
    for row in old['physics']['transport_cases']:
        means=np.array(row['means']);initial=means+np.array(row['amplitudes'])*np.cos(z[:,None]*np.array(row['modes'])[None,:]);m=bad.TransportModel();m.diffusivity=row['diffusivity'];y=m.predict(t,x,initial);species=bad.equilibrium(y.reshape(-1,2)).reshape(len(t),len(x),4)
        mass=float(max(abs(np.trapezoid(y,x,axis=1)/.001-means).ravel()));density=np.sum(species*(np.log(np.maximum(species,1e-300))-1),axis=2)-species[:,:,3]*np.log(.5);free_energy=np.trapezoid(density,x,axis=1);increase=float(max(np.diff(free_energy)));charge=float(max(abs(species[:,:,0]+species[:,:,1]-species[:,:,2]).ravel()))
        assert mass<1e-9 and increase<1e-9 and charge<1e-12 and species.min()>=0
        checks.append(dict(mass_error=mass,charge_error=charge,largest_free_energy_increment=increase,minimum_concentration=float(species.min())))
    g=good.TransportModel();g.diffusivity=1e-9;clean=np.array([g._binary(data['t'],data['x'],initial) for initial in data['initial']]);rng=np.random.default_rng(6112026);fits=[]
    for _ in range(256):
        sample=dict(data,concentration=clean+rng.normal(size=clean.shape)*data['sigma']);a=good.TransportModel().fit(sample);b=bad.TransportModel().fit(sample);pred=np.array([a._binary(data['t'],data['x'],initial) for initial in data['initial']]);res=(pred-sample['concentration'])/data['sigma'];chi=float(np.sum(res**2))/(res.size-1)
        assert chi<1.5 and abs(a.diffusivity/1e-9-1)<.05 and abs(a.diffusivity-b.diffusivity)<1e-20;fits.append(dict(parameter=a.diffusivity,chi2=chi))
    params=np.array([r['parameter'] for r in fits]);sensitivity=[]
    for d in np.quantile(params,np.linspace(0,1,9)):
        a=good.TransportModel();b=bad.TransportModel();a.diffusivity=b.diffusivity=float(d);ae=score(a,cases);be=score(b,cases);assert max(ae)<.05 and min(be)>.05;sensitivity.append(dict(parameter=float(d),oracle=ae,shortcut=be))
    report=dict(revision=9,calibration_regenerated=False,controls=controls,physics=dict(inherited_validation='predecessors/reaction-diffusion-r8/validation.json',inherited_sha256=sha(A/'validation.json'),unchanged_oracle_sha256=sha(T/'solution/model.py'),shortcut_independent_checks=independent,shortcut_conservation_checks=checks),noise=dict(seed=6112026,trials=256,method='256 actual calibration fits; all hidden predictions at nine fitted-parameter quantiles including extrema, not 256 full hidden predictions.',fit_min=float(params.min()),fit_max=float(params.max()),max_chi2=max(r['chi2'] for r in fits),hidden_sensitivity=sensitivity),seconds=time.monotonic()-tick,source_sha256={str(p.relative_to(T)):sha(p) for p in T.rglob('*') if p.is_file() and not any(x in p.parts for x in ['__pycache__','.pytest_cache'])})
    (S/'results/reaction-diffusion-r9-validation.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',round(report['seconds'],1),flush=True)
if __name__=='__main__':main()
