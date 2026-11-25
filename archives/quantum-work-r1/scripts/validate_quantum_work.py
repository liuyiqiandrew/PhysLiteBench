"""Validate quantum-work calibration and independently counted work cumulants."""
import argparse,hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import expm
ROOT=Path(__file__).resolve().parents[1];TASK=ROOT/'tasks/quantum-work'
def load(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def physical_checks(good,bad,ref):
 rng=np.random.default_rng(190091);unitarity=[];first_two=[];third_identity=[];jarzynski=[];gauge=[];minimum_mean=[];reference=[];refinement=[]
 for n in range(48):
  s=rng.uniform(.8,1.2);e=ref.experiment(rng.uniform(.35,1.4),rng.uniform(-1.1,1.1),rng.uniform(-1.1,1.1),rng.uniform(-np.pi,np.pi),rng.uniform(0,1.3),rng.uniform(0,1.3))
  args=[e[k] for k in ['temperature','amplitude_a','amplitude_b','phase','time_a','time_b']]
  energies,p,u=good.state_and_evolution(*args,s);rho=np.diag(p);H=np.diag(energies);B=u.conj().T@H@u
  a=good.statistics(*args,s);b=bad.statistics(*args,s)
  unitarity.append(float(max(abs(u.conj().T@u-np.eye(3)).ravel())));first_two.append(float(max(abs(a[:2]-b[:2]))))
  correction=float(np.trace(rho@(B@H@B-B@B@H)).real)
  pair=sum((p[i]-p[j])*(energies[j]-energies[i])*abs(B[i,j])**2 for i in range(3) for j in range(i+1,3))
  third_identity.extend([abs(a[2]-b[2]-correction),abs(correction-pair)])
  delta=energies[:,None]-energies[None,:];P=abs(u)**2*p[None,:]
  jarzynski.append(abs(float(np.sum(P*np.exp(-delta/e['temperature'])))-1));minimum_mean.append(a[0])
  shifted=H+.73*np.eye(3)
  first=shifted+e['amplitude_a']*good.X;second=shifted+e['amplitude_b']*(np.cos(e['phase'])*good.X+np.sin(e['phase'])*good.Y)
  ug=expm(-1j*e['time_b']*second)@expm(-1j*e['time_a']*first)
  gauge.append(float(max(abs(ug.conj().T@shifted@ug-shifted-(B-H)).ravel())))
  if n<12:
   c=ref.cumulants(*args,s);fine=ref.cumulants(*args,s,.12,48)
   reference.append(float(max(abs(a-c))));refinement.append(float(max(abs(c-fine))))
 assert max(unitarity+first_two+third_identity+jarzynski+gauge)<1e-11
 assert min(minimum_mean)>-1e-12 and max(reference+refinement)<1e-8
 zero=[]
 for s,T in [(.8,.35),(1.06,.7),(1.2,1.4)]:
  args=[T,0.,0.,.7,.8,1.1]
  zero.extend(abs(good.statistics(*args,s)));zero.extend(abs(bad.statistics(*args,s)));zero.extend(abs(ref.cumulants(*args,s)))
 assert max(zero)<1e-8
 hidden=sum(ref.hidden_inputs().values(),[]);ref.propagate.cache_clear();ref.cumulants.cache_clear();start=time.perf_counter();truth=ref.predict(hidden);clock=time.perf_counter()-start
 error=float(max(abs(good.predict_at(hidden,ref.TRUE_PARAMETER)-truth)));assert error<1e-8 and clock<30
 third_inputs=[e for e in hidden if e['cumulant']==3]
 skew=[];third=[]
 for e in third_inputs:
  a=good.statistics(*[e[k] for k in ['temperature','amplitude_a','amplitude_b','phase','time_a','time_b']],ref.TRUE_PARAMETER)
  third.append(abs(a[2]));skew.append(abs(a[2])/a[1]**1.5)
 assert min(third)>.1 and min(skew)>.45
 calibration=ref.calibration_inputs();difference=float(max(abs(good.predict_at(calibration,1.06)-bad.predict_at(calibration,1.06))))
 assert difference<1e-13
 grid=np.linspace(.8,1.2,121);profiles=[];fits=[]
 for s in [.801,1.06,1.199]:
  exact=good.predict_at(calibration,s);loss=np.array([np.mean((good.predict_at(calibration,z)-exact)**2) for z in grid]);idx=int(np.argmin(loss))
  assert np.all(np.diff(loss[:idx+1])<0) and np.all(np.diff(loss[idx:])>0)
  rs=[dict(input=e,value=float(v),sigma=.0003) for e,v in zip(calibration,exact)]
  fitted=good.Model().fit(rs).energy_scale;fits.append(abs(fitted/s-1));profiles.append(dict(true_scale=s,profile_minimum=float(grid[idx]),unique_profile_minimum=True))
 assert max(fits)<1e-6
 for e in calibration+hidden:
  assert .35<=e['temperature']<=1.4 and -1.1<=e['amplitude_a']<=1.1 and -1.1<=e['amplitude_b']<=1.1
  assert -np.pi<=e['phase']<=np.pi and 0<=e['time_a']<=1.3 and 0<=e['time_b']<=1.3 and e['cumulant'] in [1,2,3]
 return dict(unitarity_error=max(unitarity),first_two_cumulant_equivalence=max(first_two),third_commutator_and_pair_identity_error=float(max(third_identity)),cyclic_Jarzynski_error=max(jarzynski),common_energy_shift_invariance=max(gauge),minimum_sampled_mean_work=min(minimum_mean),off_grid_reference_error=max(reference),Cauchy_contour_refinement=max(refinement),zero_drive_work_error=float(max(zero)),hidden_reference_error=error,cold_hidden_reference_seconds=clock,hidden_minimum_absolute_third=min(third),hidden_minimum_standardized_third=min(skew),calibration_equivalence=difference,noiseless_fit_errors=fits,full_range_identifiability_profiles=profiles,public_input_domain_verified=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--generate',action='store_true');p.add_argument('--noise-trials',type=int,default=256);args=p.parse_args();start=time.time()
 good=load(TASK/'solution/model.py','oracle');bad=load(ROOT/'scripts/quantum_work_baseline.py','shortcut');ref=load(TASK/'tests/reference.py','reference')
 meta=json.loads((TASK/'tests/metadata.json').read_text());sigma=meta['measurement_sigma'];true=ref.TRUE_PARAMETER;inputs=ref.calibration_inputs();exact=ref.predict(inputs)
 if args.generate:
  values=exact+np.random.default_rng(meta['calibration_seed']).normal(0,sigma,len(inputs));records=[dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,values)]
  for directory in ['environment','tests']:(TASK/directory/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
 records=json.loads((TASK/'environment/data/calibration.json').read_text());assert records==json.loads((TASK/'tests/data/calibration.json').read_text())
 assert [r['input'] for r in records]==inputs and all(r['sigma']==sigma for r in records)
 hidden=ref.hidden_inputs();truth={k:ref.predict(es) for k,es in hidden.items()};limit=meta['prediction_limit'];diagnostic=[k for k in hidden if k!='lower_cumulants']
 def chi(m,rs):return float(np.sum(((m.predict(inputs)-np.array([r['value'] for r in rs]))/sigma)**2)/(len(rs)-1))
 def score(m):return {k:float(np.sqrt(np.mean((m.predict(es)-truth[k])**2)/np.mean(truth[k]**2))) for k,es in hidden.items()}
 report=dict(task='quantum-work',revision=1,noise_trials=args.noise_trials,calibration_seed=meta['calibration_seed'],noise_seed=meta['noise_seed'],controls={})
 for label,source in [('oracle',good),('shortcut',bad)]:
  m=source.Model().fit(records);q=score(m);r=dict(parameter=m.energy_scale,parameter_relative_error=abs(m.energy_scale/true-1),calibration_chi2=chi(m,records),hidden=q)
  assert r['calibration_chi2']<1.5 and r['parameter_relative_error']<.03 and q['lower_cumulants']<limit
  assert (max(q.values())<limit if label=='oracle' else min(q[k] for k in diagnostic)>limit)
  report['controls'][label]=r
 rng=np.random.default_rng(meta['noise_seed']);parameters=[];cal=[];oracle=[];shortcut=[]
 for _ in range(args.noise_trials):
  rs=[dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,exact+rng.normal(0,sigma,len(inputs)))];a=good.Model().fit(rs);b=bad.Model().fit(rs)
  assert abs(a.energy_scale-b.energy_scale)<1e-7
  parameters.append(a.energy_scale);cal.append(chi(a,rs));oracle.append(max(score(a).values()));q=score(b);shortcut.append(min(q[k] for k in diagnostic));assert q['lower_cumulants']<limit
  assert cal[-1]<1.5 and abs(a.energy_scale/true-1)<.03 and oracle[-1]<limit and shortcut[-1]>limit
 report['noise']=dict(calibration_passes=args.noise_trials,parameter_passes=args.noise_trials,oracle_passes=args.noise_trials,shortcut_rejections=args.noise_trials,maximum_chi2=max(cal),parameter_min=min(parameters),parameter_max=max(parameters),oracle_hidden_max=max(oracle),shortcut_discriminating_min=min(shortcut))
 extrema={}
 for name,source in [('oracle',good),('shortcut',bad)]:
  errors=[]
  for s in [min(parameters),max(parameters)]:
   m=source.Model();m.energy_scale=s;q=score(m);errors.extend(q[k] for k in diagnostic)
  extrema[name]=dict(min=min(errors),max=max(errors))
 assert extrema['oracle']['max']<limit and extrema['shortcut']['min']>limit
 report['noise_extrema']=extrema;report['physical_checks']=physical_checks(good,bad,ref);report['seconds']=time.time()-start
 paths=[TASK/n for n in ['instruction.md','environment/README.md','environment/model.py','solution/model.py','tests/reference.py','environment/data/calibration.json','tests/metadata.json']]+[ROOT/'scripts/quantum_work_baseline.py',Path(__file__)]
 report['source_sha256']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
 output=ROOT/'jobs/quantum-work-validation';output.mkdir(parents=True,exist_ok=True)
 for dest in [output/'summary.json',ROOT/'results/quantum-work-validation.json']:dest.write_text(json.dumps(report,indent=2)+'\n')
 print('quantum-work PASS',report['seconds']);print(json.dumps(report['controls'],indent=2));print(json.dumps(report['noise'],indent=2));print(json.dumps(report['physical_checks'],indent=2))
if __name__=='__main__':main()
