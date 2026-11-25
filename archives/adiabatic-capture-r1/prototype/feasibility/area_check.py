"""Bounded static area/period check; no trajectory convergence or task validation."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

def v(q,delta):return q**4/4-q*q/2+delta*q

def geometry(delta):
 x=np.sort(np.roots([1,0,-1,delta]).real)
 saddle=x[1];barrier=v(saddle,delta)
 b=np.sqrt(2*(1-saddle*saddle))
 left,right=-saddle-b,-saddle+b
 def p(q):return abs(q-saddle)*np.sqrt(max(0.,(right-q)*(q-left)))/np.sqrt(2)
 a=np.array([2*quad(p,left,saddle,epsabs=1e-12)[0],2*quad(p,saddle,right,epsabs=1e-12)[0]])
 return x,barrier,a

def intervals(e,delta):
 roots=np.roots([.25,0,-.5,delta,-e]);r=np.sort(roots.real[abs(roots.imag)<1e-8])
 return list(zip(r[::2],r[1::2]))

def area_period(e,delta,branch=None):
 aa=tt=0.
 for lo,hi in intervals(e,delta):
  if branch is not None and (hi<geometry(delta)[0][1])!=(branch==0):continue
  center=(hi+lo)/2;half=(hi-lo)/2
  def integrand(theta,kind):
   q=center+half*np.sin(theta);jac=half*np.cos(theta)
   # Product form avoids cancellation near turning points.
   other=np.roots([.25,0,-.5,delta,-e]);gap=(-.25*np.prod(q-other)).real
   p=np.sqrt(max(gap,1e-300)*2)
   return 2*jac*(p if kind==0 else 1/p)
  aa+=quad(lambda z:integrand(z,0),-np.pi/2,np.pi/2,epsabs=2e-10)[0]
  tt+=quad(lambda z:integrand(z,1),-np.pi/2,np.pi/2,epsabs=2e-9)[0]
 return aa,tt

def energy_for_area(a,delta,branch=None):
 x,barrier,sep=geometry(delta)
 low=min(v(x[[0,2]],delta)) if branch is None else v(x[0 if branch==0 else 2],delta)
 lo=low+1e-9
 hi=barrier-1e-9 if branch is not None else max(1.,barrier+1.)
 while area_period(hi,delta,branch)[0]<a:hi=2*hi+1
 return brentq(lambda e:area_period(e,delta,branch)[0]-a,lo,hi,xtol=2e-11)

rows=[]
for delta in [.15,.2,.25]:
 x,barrier,lobes=geometry(delta);prob=lobes/sum(lobes)
 for action in [.12,.16,.2]:
  s_final=2.;a=2*np.pi*action/s_final**1.5
  energies=[];kinetic=[]
  for side in [0,1]:
   e=energy_for_area(prob[side]*a,delta,side)
   _,period=area_period(e,delta,side)
   energies.append(s_final*s_final*e);kinetic.append(s_final*s_final*prob[side]*a/(2*period))
  es=energy_for_area(a,delta);_,ts=area_period(es,delta)
  source=s_final*s_final*a/(2*ts);physical=float(prob@kinetic)
  rows.append({'tilt':delta,'initial_action':action,'final_scale':s_final,'lobe_capture_fractions':prob.tolist(),'initial_scale_separatrix_action':float(sum(lobes)*.05**1.5/(2*np.pi)),'largest_calibration_endpoint_separatrix_action':float(sum(lobes)*.2**1.5/(2*np.pi)),'final_separatrix_action':float(sum(lobes)*s_final**1.5/(2*np.pi)),'physical_kinetic_area_prediction':physical,'global_phase_volume_kinetic':source,'relative_gap':abs(source/physical-1),'physical_branch_energies':energies,'source_common_energy':s_final*s_final*es})
assert all(np.isfinite(x['relative_gap']) and x['largest_calibration_endpoint_separatrix_action']<.12 and x['final_separatrix_action']>.2 for x in rows)
report={'status':'bounded_static_feasibility_only','candidate_hamiltonian':'H=p^2/2+q^4/4-s*q^2/2+delta*s^(3/2)*q','source':'Preserve total enclosed phase volume and mix the complete final energy shell.','physical_area_construction':'Both lobe areas grow as s^(3/2); capture fractions are the normalized lobe areas, and each captured branch inherits that fraction of the incoming enclosed area.','readout':'Final ensemble mean kinetic energy, averaged over final orbital phase.','rows':rows,'minimum_relative_gap':min(x['relative_gap'] for x in rows),'minimum_physical_signal':min(x['physical_kinetic_area_prediction'] for x in rows),'maximum_physical_signal':max(x['physical_kinetic_area_prediction'] for x in rows),'qualification':'Static area predictions only. Initial finite-width action-band averaging, full-domain controls, Hamiltonian-trajectory slow-ramp convergence, uncertainty and grading have not been run. This is not an independent dynamical verification or a task package.'}
Path(__file__).with_name('area-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
