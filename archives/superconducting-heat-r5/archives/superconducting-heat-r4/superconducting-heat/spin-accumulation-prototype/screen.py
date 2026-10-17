from pathlib import Path
import importlib.util,json,time
import numpy as np
p=Path('/private/tmp/physlite-super-spin-accumulation-prototype.py');s=p.read_text().replace('upper=np.sqrt(80*max(tl,tr)+w)','upper=np.sqrt((60*max(tl,tr)) if not a else (80*max(tl,tr)+w))');p.write_text(s)
spec=importlib.util.spec_from_file_location('proto',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
start=time.time();ref=m.ref3
corners=[ref.experiment(l,r,p,a,w) for l in (.25,1.05) for r in (.25,1.05) for p in (0.,np.pi,-.7) for a in (0.,.4,1.) for w in (.1,.5,7.,12.)]
rng=np.random.default_rng(63021)
interiors=[ref.experiment(rng.uniform(.25,1.05),rng.uniform(.25,1.05),rng.uniform(-np.pi,np.pi),rng.uniform(0,1),rng.uniform(*((.1,.5) if i%2 else (7,12)))) for i in range(32)]
cases=sum(ref.hidden_inputs().values(),[])+corners+interiors
report={'domain_settings':len(cases),'calibration_max_pW':0.,'energy_residual_max_pW':0.,'spin_work_residual_max_pW':0.,'stationarity_relative_max':0.,'entropy_min_W_per_K':float('inf'),'column_dominance_min':float('inf'),'maximum_mu_K':0.,'maximum_abs_mu_over_drive':0.,'multiple_start_error_K':0.,'spinflip_micro_macro_checks':[]}
for i,e in enumerate(cases):
 tl,tr,p,a,w=[e.get(k,d) for k,d in zip(('left_temperature','right_temperature','phase','amplitude','drive_energy'),(None,None,None,0.,.3))]
 prep=m.preparation(tl,tr,p,a,w);v=m.steady(prep);mu=v['mu_K'];J=v['jacobian']
 report['maximum_mu_K']=max(report['maximum_mu_K'],float(max(abs(mu))));report['maximum_abs_mu_over_drive']=max(report['maximum_abs_mu_over_drive'],float(max(abs(mu))/w))
 report['energy_residual_max_pW']=max(report['energy_residual_max_pW'],v['energy_work_residual_pW']);report['spin_work_residual_max_pW']=max(report['spin_work_residual_max_pW'],v['spin_work_identity_pW'])
 report['stationarity_relative_max']=max(report['stationarity_relative_max'],float(max(abs(v['stationarity_residual_per_s']))/(m.RATE*45.)))
 report['entropy_min_W_per_K']=min(report['entropy_min_W_per_K'],v['tunneling_entropy']+v['spinflip_entropy'])
 report['column_dominance_min']=min(report['column_dominance_min'],float(min(-J[0,0]-abs(J[1,0]),-J[1,1]-abs(J[0,1]))))
 if i>=len(cases)-len(interiors):
  for initial in ((w/3,w/3),(-w/3,-w/3),(w/2,-w/2)):
   other=m.steady(prep,start=initial);report['multiple_start_error_K']=max(report['multiple_start_error_K'],float(max(abs(other['mu_K']-mu))))
  for point in (np.array([0.,0.]),np.array([w/3,-w/3])):
   jac=m.evaluate(prep,point)[2]-m.NU/(m.TAU*m.RATE*45.)*np.diag([m.relaxation(t,d,x)[1] for t,d,x in zip((tl,tr),prep[2:4],point)])
   assert min(-jac[0,0]-abs(jac[1,0]),-jac[1,1]-abs(jac[0,1]))>0
for e in ref.calibration_inputs():
 prep=m.preparation(e['left_temperature'],e['right_temperature'],e['phase'],0.,.3);v=m.steady(prep)
 assert max(abs(v['mu_K']))==0
 report['calibration_max_pW']=max(report['calibration_max_pW'],float(abs(v['heat_pW'][0]-m.r3.predict_at([e],45.)[0])))
# Explicit microscopic spinflip rates cancel the positive-energy BCS DOS.
from scipy.integrate import quad
for t,d,mu in ((.25,2.,1.),(.8,3.,2.),(.9,1.,-.3)):
 def integrand(z):
  E=d+z*z;dos=E/np.sqrt((E-d)*(E+d));rate=np.sqrt((E-d)*(E+d))/E/m.TAU
  return 2*z*dos*rate*(m.expit((mu-E)/t)-m.expit((-mu-E)/t))
 integral=quad(integrand,0,np.sqrt(80*t+abs(mu)),epsabs=1e-6,epsrel=1e-12)[0]*m.NU
 closed=m.NU/m.TAU*m.relaxation(t,d,mu)[0]
 report['spinflip_micro_macro_checks'].append({'temperature_K':t,'gap_K':d,'mu_K':mu,'relative_rate_error':float(abs(integral-closed)/max(abs(closed),1.))})
report['fast_relaxation_limit']={}
for tau in (1e-10,1e-12):
 errors=[];mus=[]
 for e in sum(ref.hidden_inputs().values(),[]):
  prep=m.preparation(e['left_temperature'],e['right_temperature'],e['phase'],e['amplitude'],e['drive_energy']);v=m.steady(prep,tau=tau);errors.append(abs(v['heat_pW'][0]-m.r3.predict_at([e],45.)[0]));mus.extend(abs(v['mu_K']))
 report['fast_relaxation_limit'][str(tau)]={'maximum_mu_K':float(max(mus)),'maximum_heat_error_pW':float(max(errors))}
assert report['calibration_max_pW']<1e-12 and report['energy_residual_max_pW']<1e-10 and report['spin_work_residual_max_pW']<1e-10
assert report['stationarity_relative_max']<1e-10 and report['entropy_min_W_per_K']>=-1e-24 and report['column_dominance_min']>0
assert report['multiple_start_error_K']<1e-9
report['seconds']=time.time()-start
out=Path('/private/tmp/physlite-super-spin-accumulation-screen.json');out.write_text(json.dumps(report,indent=2)+'\n');print('DOMAIN',json.dumps(report,indent=2))
