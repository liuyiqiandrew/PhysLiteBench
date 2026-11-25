import json,time
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh
from numpy.polynomial.legendre import leggauss

def oscillator_capacity(w,T):
 x=np.asarray(w)/T;z=np.exp(-x)
 return np.where(x==0,1.,x*x*z/(-np.expm1(-x))**2)

def denominator(w,w0,gamma,cutoff):
 return w0*w0-w*w-1j*w*gamma*cutoff/(cutoff-1j*w)

def susceptibility(w,w0,gamma,cutoff):return 1/denominator(w,w0,gamma,cutoff)

def spectral_shift(w,w0,gamma,cutoff):
 d=denominator(w,w0,gamma,cutoff)
 derivative=-2*w-1j*gamma*cutoff**2/(cutoff-1j*w)**2
 return float(np.imag(-derivative/d)/np.pi)

def capacity(T,w0,gamma,cutoff,local=False):
 if gamma==0:return float(oscillator_capacity(w0,T))
 def integrand(w):
  c=float(oscillator_capacity(w,T))
  if local:
   if w==0:weight=gamma/(np.pi*w0*w0)
   else:weight=(w*w+w0*w0)*susceptibility(w,w0,gamma,cutoff).imag/(np.pi*w)
  else:weight=spectral_shift(w,w0,gamma,cutoff)
  return weight*c
 return quad(integrand,0,max(60*T,20*w0),epsabs=1e-11,epsrel=1e-11,points=[w0],limit=300)[0]

def finite_bath(T,w0,gamma,cutoff,order=20,cutoff_factor=32,include_local=False):
 maximum=cutoff_factor*max(cutoff,w0,gamma)
 edges=[0.,min(T,w0,cutoff)/4]
 while edges[-1]<maximum:edges.append(min(2*edges[-1],maximum))
 n,w=leggauss(order);omega=np.concatenate([(a+b)/2+(b-a)*n/2 for a,b in zip(edges[:-1],edges[1:])]);weights=np.concatenate([(b-a)*w/2 for a,b in zip(edges[:-1],edges[1:])])
 spectral=gamma*omega*cutoff**2/(cutoff**2+omega**2)
 coupling=np.sqrt((2/np.pi)*spectral*omega*weights)
 stiffness=np.diag(np.r_[w0*w0+np.sum((coupling/omega)**2),omega**2]);stiffness[0,1:]=-coupling;stiffness[1:,0]=-coupling
 if include_local:values,vectors=eigh(stiffness,check_finite=False)
 else:values=eigh(stiffness,eigvals_only=True,check_finite=False)
 assert values.min()>-1e-8
 full=oscillator_capacity(np.sqrt(np.maximum(values,0)),T)
 result=(float(np.sum(full)-np.sum(oscillator_capacity(omega,T))),len(omega),float(values.min()))
 if include_local:
  local=float(np.sum(vectors[0]**2*(1+w0*w0/values)*full)/2)
  return result+(local,)
 return result

if __name__=='__main__':
 start=time.time();rows=[]
 for g in [1.,2.,3.]:
  for cutoff in [1.,2.,4.]:
   for T in [.1,.2,.35,.6,1.]:
    truth=capacity(T,1.04,g,cutoff);local=capacity(T,1.04,g,cutoff,True)
    rows.append(dict(temperature=T,gamma=g,cutoff=cutoff,excess_capacity=truth,local_capacity=local,relative_error=abs(local/truth-1)))
 checks=[]
 for T,w0,g,cutoff in [(.35,1.04,3.,1.),(.6,1.04,2.,2.),(1.,1.04,3.,4.),(.1,.8,1.,4.)]:
  truth=capacity(T,w0,g,cutoff);a,n,_=finite_bath(T,w0,g,cutoff,16);b,n2,minv=finite_bath(T,w0,g,cutoff,24);c,n3,_=finite_bath(T,w0,g,cutoff,24,64)
  checks.append(dict(T=T,w0=w0,gamma=g,cutoff=cutoff,phase=truth,finite16=a,finite24=b,finite24_extended=c,basis_refinement=abs(a-b),tail_refinement=abs(b-c),final_error=abs(c-truth),mode_counts=[n,n2,n3],min_stiffness=minv))
 local_checks=[]
 for T,w0,g,cutoff in [(.35,1.04,3.,1.),(.6,1.04,2.,2.),(1.,1.04,3.,4.)]:
  local=capacity(T,w0,g,cutoff,True)
  a=finite_bath(T,w0,g,cutoff,24,32,True)[3]
  b=finite_bath(T,w0,g,cutoff,24,64,True)[3]
  local_checks.append(dict(T=T,w0=w0,gamma=g,cutoff=cutoff,susceptibility_capacity=local,finite_bath_local_capacity=b,tail_refinement=abs(a-b),final_error=abs(b-local)))
 report=dict(status='prototype only',rows=rows,independent_finite_bath_checks=checks,independent_local_energy_checks=local_checks,max_relative_error=max(x['relative_error'] for x in rows),primary_source='https://arxiv.org/abs/0811.3509',seconds=time.time()-start)
 Path('results/quantum-calorimetry-prototype.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
