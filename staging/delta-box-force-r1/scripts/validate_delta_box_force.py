"""Validate delta-box-force calibration and independently computed wall forces."""
import argparse,hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];TASK=ROOT/'tasks/delta-box-force'
def load(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def physical_checks(good,bad,ref):
    from scipy.optimize import minimize_scalar
    from scipy.integrate import quad
    comparison=[];refinement=[];displacement=[];tails=[];gaps=[];normalization=[];jump=[];virial=[];hellmann=[]
    def free_energy(L,g,T,m):
        e=bad.levels(L,g,m)
        return float(e[0]-T*np.log(np.sum(np.exp(-(e-e[0])/T))))
    for m in [.4,.63,.9]:
        for L in [.8,1.8]:
            for g in [0.,3.,12.]:
                for T in [.1,2.]:
                    e=ref.experiment(L,g,T);truth=good.force(e,m);reference=ref.force(e,m);fine=ref.force(e,m,2047)
                    comparison.append(abs(truth-reference));refinement.append(abs(reference-fine))
                    h=2e-4*L
                    derivative=-(free_energy(L-2*h,g,T,m)-8*free_energy(L-h,g,T,m)+8*free_energy(L+h,g,T,m)-free_energy(L+2*h,g,T,m))/(12*h)
                    displacement.append(abs(derivative-truth))
                    en,fn=good.levels(L,g,m,24);p=np.exp(-(en-en[0])/T);p/=p.sum();tails.append(abs(p@fn-truth));gaps.append(float(min(np.diff(en))))
    for L,g,m in [(1.,2.,.4),(1.5,4.,.63),(1.8,12.,.9)]:
        a=1/(2*m);ratio=g*L/(4*a)
        for n in range(4):
            x=good.brentq(lambda x:x/np.tan(x)+ratio,(n+.5)*np.pi+1e-12,(n+1)*np.pi-1e-12,xtol=2e-14)
            k=2*x/L;norm=L/2-np.sin(k*L)/(2*k);amp=1/np.sqrt(norm);E=a*k*k;contact=amp**2*np.sin(x)**2;F=a*k*k/norm
            normalization.append(abs(2*quad(lambda y:amp**2*np.sin(k*(L/2-y))**2,0,L/2,epsabs=1e-12)[0]-1))
            jump.append(abs(-2*amp*k*np.cos(x)-2*m*g*amp*np.sin(x)))
            virial.append(abs((2*E-g*contact)/L-F))
            h=2e-4*g
            def energy(gg):
                xx=good.brentq(lambda x:x/np.tan(x)+gg*L/(4*a),(n+.5)*np.pi+1e-12,(n+1)*np.pi-1e-12,xtol=2e-14)
                return a*(2*xx/L)**2
            derivative=(energy(g-2*h)-8*energy(g-h)+8*energy(g+h)-energy(g+2*h))/(12*h)
            hellmann.append(abs(derivative-contact))
    assert max(comparison+refinement)<1e-6 and max(displacement)<1e-7 and max(tails)<1e-10
    assert min(gaps)>0 and max(normalization+jump+virial+hellmann)<1e-7
    tiny=[]
    for mass in [.4,.63,.9]:
        base=ref.experiment(1.3,0.,.7)
        for strength in [1e-18,1e-15,1e-12,1e-9]:
            e=dict(base,strength=strength)
            tiny.extend([abs(good.force(e,mass)-good.force(base,mass)),abs(bad.force(e,mass)-bad.force(base,mass))])
    assert max(tiny)<1e-7
    hidden=sum(ref.hidden_inputs().values(),[]);ref.modes.cache_clear();start=time.perf_counter();truth=ref.predict(hidden);clock=time.perf_counter()-start
    hidden_error=float(max(abs(good.predict_at(hidden,ref.TRUE_PARAMETER)-truth)));assert hidden_error<1e-6 and clock<30
    cal=ref.calibration_inputs();equivalence=float(max(abs(good.predict_at(cal,.63)-bad.predict_at(cal,.63))));assert equivalence<1e-12
    profiles=[];fits=[];grid=np.linspace(.4,.9,121)
    for mass in [.401,.63,.899]:
        exact=good.predict_at(cal,mass);loss=np.array([np.mean((good.predict_at(cal,z)-exact)**2) for z in grid]);idx=int(np.argmin(loss))
        assert np.all(np.diff(loss[:idx+1])<0) and np.all(np.diff(loss[idx:])>0)
        rs=[dict(input=e,value=float(v),sigma=.001) for e,v in zip(cal,exact)];fit=good.Model().fit(rs).mass;fits.append(abs(fit/mass-1));profiles.append(dict(true_mass=mass,profile_minimum=float(grid[idx]),unique_profile_minimum=True))
    assert max(fits)<1e-6
    for e in cal+hidden:assert .8<=e['length']<=1.8 and 0<=e['strength']<=12 and .1<=e['temperature']<=2
    return dict(full_domain_corner_oracle_reference_error=float(max(comparison)),reference_1023_to_2047_Richardson_refinement=float(max(refinement)),fixed_strength_free_energy_displacement_error=float(max(displacement)),thermal_32_to_48_level_error=float(max(tails)),minimum_sampled_spectral_gap=min(gaps),wavefunction_normalization_error=max(normalization),defect_derivative_jump_error=max(jump),contact_virial_identity_error=max(virial),contact_Hellmann_Feynman_error=max(hellmann),hidden_reference_error=hidden_error,cold_hidden_reference_seconds=clock,calibration_equivalence=equivalence,noiseless_mass_recovery_errors=fits,full_range_mass_identifiability=profiles,public_input_domain_verified=True,near_zero_strength_continuity_error=float(max(tiny)))

def main():
 p=argparse.ArgumentParser();p.add_argument('--generate',action='store_true');p.add_argument('--noise-trials',type=int,default=256);args=p.parse_args();start=time.time()
 good=load(TASK/'solution/model.py','oracle');bad=load(ROOT/'scripts/delta_box_force_baseline.py','shortcut');ref=load(TASK/'tests/reference.py','reference')
 meta=json.loads((TASK/'tests/metadata.json').read_text());sigma=meta['measurement_sigma'];true=ref.TRUE_PARAMETER;inputs=ref.calibration_inputs();exact=ref.predict(inputs)
 if args.generate:
  values=exact+np.random.default_rng(meta['calibration_seed']).normal(0,sigma,len(inputs));records=[dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,values)]
  for directory in ['environment','tests']:(TASK/directory/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
 records=json.loads((TASK/'environment/data/calibration.json').read_text());assert records==json.loads((TASK/'tests/data/calibration.json').read_text())
 assert [r['input'] for r in records]==inputs and all(r['sigma']==sigma for r in records)
 hidden=ref.hidden_inputs();truth={k:ref.predict(es) for k,es in hidden.items()};limit=meta['prediction_limit'];diagnostic=[k for k in hidden if k!='free_anchors']
 def chi(m,rs):return float(np.sum(((m.predict(inputs)-np.array([r['value'] for r in rs]))/sigma)**2)/(len(rs)-1))
 def score(m):return {k:float(np.sqrt(np.mean((m.predict(es)-truth[k])**2)/np.mean(truth[k]**2))) for k,es in hidden.items()}
 report=dict(task='delta-box-force',revision=1,noise_trials=args.noise_trials,calibration_seed=meta['calibration_seed'],noise_seed=meta['noise_seed'],controls={})
 for label,source in [('oracle',good),('shortcut',bad)]:
  m=source.Model().fit(records);q=score(m);r=dict(parameter=m.mass,parameter_relative_error=abs(m.mass/true-1),calibration_chi2=chi(m,records),hidden=q)
  assert r['calibration_chi2']<1.5 and r['parameter_relative_error']<.03 and q['free_anchors']<limit
  assert (max(q.values())<limit if label=='oracle' else min(q[k] for k in diagnostic)>limit)
  report['controls'][label]=r
 rng=np.random.default_rng(meta['noise_seed']);parameters=[];cal=[];oracle=[];shortcut=[]
 for _ in range(args.noise_trials):
  rs=[dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,exact+rng.normal(0,sigma,len(inputs)))];a=good.Model().fit(rs);b=bad.Model().fit(rs)
  assert abs(a.mass-b.mass)<1e-7
  parameters.append(a.mass);cal.append(chi(a,rs));oracle.append(max(score(a).values()));q=score(b);shortcut.append(min(q[k] for k in diagnostic));assert q['free_anchors']<limit
  assert cal[-1]<1.5 and abs(a.mass/true-1)<.03 and oracle[-1]<limit and shortcut[-1]>limit
 report['noise']=dict(calibration_passes=args.noise_trials,parameter_passes=args.noise_trials,oracle_passes=args.noise_trials,shortcut_rejections=args.noise_trials,maximum_chi2=max(cal),parameter_min=min(parameters),parameter_max=max(parameters),oracle_hidden_max=max(oracle),shortcut_discriminating_min=min(shortcut))
 extrema={}
 for name,source in [('oracle',good),('shortcut',bad)]:
  errors=[]
  for s in [min(parameters),max(parameters)]:
   m=source.Model();m.mass=s;q=score(m);errors.extend(q[k] for k in diagnostic)
  extrema[name]=dict(min=min(errors),max=max(errors))
 assert extrema['oracle']['max']<limit and extrema['shortcut']['min']>limit
 report['noise_extrema']=extrema;report['physical_checks']=physical_checks(good,bad,ref);report['seconds']=time.time()-start
 paths=[TASK/n for n in ['instruction.md','environment/README.md','environment/model.py','solution/model.py','tests/reference.py','environment/data/calibration.json','tests/metadata.json']]+[ROOT/'scripts/delta_box_force_baseline.py',Path(__file__)]
 report['source_sha256']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
 output=ROOT/'jobs/delta-box-force-validation';output.mkdir(parents=True,exist_ok=True)
 for dest in [output/'summary.json',ROOT/'results/delta-box-force-validation.json']:dest.write_text(json.dumps(report,indent=2)+'\n')
 print('delta-box-force PASS',report['seconds']);print(json.dumps(report['controls'],indent=2));print(json.dumps(report['noise'],indent=2));print(json.dumps(report['physical_checks'],indent=2))
if __name__=='__main__':main()
