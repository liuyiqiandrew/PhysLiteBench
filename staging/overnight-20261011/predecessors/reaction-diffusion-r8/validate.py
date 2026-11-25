"""Validate rapid neutral association against independently transported species."""
from pathlib import Path
import hashlib,importlib.util,json,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];TASK=ROOT/'tasks/reaction-diffusion'
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def hidden_score(model,cases):
    scores=[]
    for t,x,initial,truth in cases:
        y=model.predict(t,x,initial);mean=np.trapezoid(initial,x,axis=0)/(x[-1]-x[0])
        scores.append(float(np.max(np.linalg.norm(y-truth,axis=(0,1))/np.linalg.norm(truth-mean,axis=(0,1)))))
    return scores

def main():
    started=time.monotonic();good=load('oracle',TASK/'solution/model.py');bad=load('shortcut',ROOT/'scripts/reaction_baseline.py');ref=load('reference',TASK/'tests/reference.py')
    assert sha(TASK/'environment/data/calibration.npz')==sha(TASK/'tests/data/calibration.npz')
    data=ref.load_data();cases=ref.hidden_cases();controls={}
    for name,m in [('oracle',good),('shortcut',bad)]:
        fit=m.TransportModel().fit(data);row=ref.metrics(fit,data)
        assert row['calibration_reduced_chi2']<1.5 and row['relative_diffusivity_error']<.05
        assert max(row['hidden_nrmse'])<.05 if name=='oracle' else min(row['hidden_nrmse'])>.05
        controls[name]=row
    print('Controls',json.dumps(controls),flush=True)
    refinement=[]
    for i,(t,x,initial,truth) in enumerate(cases):
        finer=ref.trajectory(t,x,initial,refinement=8,rate=40.)
        rate_fine=ref.trajectory(t,x,initial,refinement=4,rate=40.)
        mean=np.trapezoid(initial,x,axis=0)/(x[-1]-x[0]);scale=np.linalg.norm(truth-mean)
        row=dict(case=i,grid_and_reaction_refinement=float(np.linalg.norm(finer-truth)/scale),reaction_refinement=float(np.linalg.norm(rate_fine-truth)/scale))
        assert row['grid_and_reaction_refinement']<.002 and row['reaction_refinement']<.0001,row
        refinement.append(row)
    rng=np.random.default_rng(2026101188);chemistry=[]
    for _ in range(128):
        total=rng.uniform(0.,3.,(20,2));species=good.equilibrium(total);reference=ref.associated_initial(total)
        assert np.max(abs(species-reference))<1e-10
        residual=species[:,3]-.5*species[:,0]*species[:,1]*species[:,2]**2
        assert np.max(abs(residual))<1e-10 and species.min()>=0
        assert np.max(abs(species[:,0]+species[:,1]-species[:,2]))<1e-12
        assert np.max(abs(species[:,:2]+species[:,3:4]-total))<1e-12
        chemistry.append(float(np.max(abs(residual))))
    physics=[];x=np.linspace(0,.001,65);z=np.pi*x/.001;t=np.linspace(0,180,19)
    for i in range(16):
        means=rng.uniform(.5,1.4,2);amps=rng.uniform(.1,.4,2);modes=rng.integers(1,4,2)
        initial=means+amps*np.cos(z[:,None]*modes[None,:]);diff=float(rng.uniform(.6e-9,1.4e-9))
        model=good.TransportModel();model.diffusivity=diff;y=model.predict(t,x,initial)
        expected=ref.trajectory(t,x,initial,diffusivity=diff,refinement=4,rate=40.)
        scale=np.linalg.norm(expected-means);err=float(np.linalg.norm(y-expected)/scale)
        mass=float(np.max(abs(np.trapezoid(y,x,axis=1)/.001-means)))
        mirrored=model.predict(t,x,initial[::-1]);mirror=float(np.max(abs(mirrored[:,::-1]-y)))
        species=good.equilibrium(y.reshape(-1,2)).reshape(len(t),len(x),4)
        density=np.sum(species*(np.log(np.maximum(species,1e-300))-1),axis=2)-species[:,:,3]*np.log(.5)
        free_energy=np.trapezoid(density,x,axis=1);increase=float(np.max(np.diff(free_energy)))
        assert err<.008 and mass<1e-9 and mirror<1e-8 and increase<1e-9 and y.min()>0
        physics.append(dict(diffusivity=diff,means=means.tolist(),amplitudes=amps.tolist(),modes=modes.tolist(),reference_nrmse=err,mass_error=mass,reflection_error=mirror,largest_free_energy_increment=increase,minimum=float(y.min())))
    # The no-association limit reproduces the predecessor's nonlinear transport.
    good.ASSOCIATION=0.;a=good.TransportModel();b=bad.TransportModel();a.diffusivity=b.diffusivity=1e-9
    t0,x0,initial,_=cases[0];dissociated=float(np.max(abs(a.predict(t0,x0,initial)-b.predict(t0,x0,initial))))
    assert dissociated<1e-7;good.ASSOCIATION=.5
    print('Independent kinetics, mass, charge, free energy and limits PASS',flush=True)
    physical=good.TransportModel();physical.diffusivity=1e-9
    clean=np.array([physical._binary(data['t'],data['x'],initial) for initial in data['initial']]);rows=[]
    rng=np.random.default_rng(6112026)
    for i in range(256):
        sample=dict(data,concentration=clean+rng.normal(size=clean.shape)*data['sigma'])
        fit=good.TransportModel().fit(sample);short=bad.TransportModel();short.diffusivity=fit.diffusivity
        pred=np.array([fit._binary(data['t'],data['x'],initial) for initial in data['initial']]);res=(pred-sample['concentration'])/data['sigma']
        oe=hidden_score(fit,cases);be=hidden_score(short,cases)
        row=dict(diffusivity=fit.diffusivity,chi2=float(np.sum(res**2))/(res.size-1),oracle_max=max(oe),shortcut_min=min(be))
        assert abs(fit.diffusivity/1e-9-1)<.05 and row['chi2']<1.5 and row['oracle_max']<.05 and row['shortcut_min']>.05
        rows.append(row)
        if (i+1)%64==0:print('Direct noisy fits and all hidden predictions',i+1,flush=True)
    report=dict(revision=8,calibration_regenerated=False,controls=controls,physics=dict(refinement=refinement,equilibrium_cases=2560,equilibrium_residual_max=max(chemistry),transport_cases=physics,dissociated_limit_error=dissociated),noise=dict(seed=6112026,trials=256,method='Direct fit and all hidden predictions for each draw',oracle_worst=max(r['oracle_max'] for r in rows),shortcut_smallest_miss=min(r['shortcut_min'] for r in rows),max_calibration_chi2=max(r['chi2'] for r in rows)),seconds=time.monotonic()-started,source_sha256={str(p.relative_to(TASK)):sha(p) for p in TASK.rglob('*') if p.is_file() and not any(x in p.parts for x in ['__pycache__','.pytest_cache'])})
    (ROOT/'results/reaction-diffusion-r8-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',seconds=report['seconds'],noise=report['noise'])),flush=True)
if __name__=='__main__':main()
