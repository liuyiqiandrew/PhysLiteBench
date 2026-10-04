"""Prototype only: local spin absorption with paired Gaussian fermions."""
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from scipy.special import expit
import json,time

N=6
ANN=[]
for a in range(N):
 op=np.zeros((2**N,2**N),complex)
 for state in range(2**N):
  if state & (1<<a):op[state^(1<<a),state]=(-1)**((state&((1<<a)-1)).bit_count())
 ANN.append(op)
NUMBER=[c.conj().T@c for c in ANN]

def matrices(e):
 hsite=np.diag(np.array([-.35,.15,.7])+e.get('offset',0.))
 for i,j,w in [(0,1,1.),(1,2,.8),(0,2,.3)]:hsite[i,j]=hsite[j,i]=-e['hopping']*w
 h=np.kron(hsite,np.eye(2));pair=np.zeros((N,N),complex)
 for i,w in enumerate([1.,.85,1.15]):
  z=e['pairing']*w*np.exp(1j*e.get('phase',0.)*[0.,1.,-.4][i]);pair[2*i,2*i+1]=z;pair[2*i+1,2*i]=-z
 return h,pair

def gaussian_lines(e,complete=True,detector=None):
 h,pair=matrices(e);B=np.block([[h,pair],[-pair.conj(),-h.T]])
 ev,V=eigh(B);occ=expit(ev/e['temperature'])
 # <Psi_a(t) Psi_b^dagger(0)> = sum_r K[a,b,r] exp(-i E_r t)
 K=np.einsum('ar,br,r->abr',V,V.conj(),occ)
 weights=np.zeros(N) if detector is None else np.array(detector,dtype=float)
 if detector is None:weights[2*e['site']:2*e['site']+2]=[.5,-.5]
 coeff=np.zeros((2*N,2*N),complex)
 for a in range(N):
  for b in range(N):
   if weights[a]*weights[b]==0:continue
   coeff+=weights[a]*weights[b]*np.outer(K[a+N,b+N],K[a,b])
   if complete:coeff-=weights[a]*weights[b]*np.outer(K[a+N,b],K[a,b+N])
 return ev[:,None]+ev[None,:],coeff.real

def fock_lines(e,detector=None):
 h,pair=matrices(e);H=np.zeros((2**N,2**N),complex)
 for a in range(N):
  for b in range(N):
   H+=h[a,b]*ANN[a].conj().T@ANN[b]
   if a<b:
    z=pair[a,b]*ANN[a].conj().T@ANN[b].conj().T;H+=z+z.conj().T
 en,V=eigh(H);p=np.exp(-(en-en[0])/e['temperature']);p/=p.sum()
 weights=np.zeros(N) if detector is None else np.array(detector,dtype=float)
 if detector is None:weights[2*e['site']:2*e['site']+2]=[.5,-.5]
 O=sum(w*n for w,n in zip(weights,NUMBER));O=V.conj().T@O@V
 # rows final n, columns initial m
 return en[:,None]-en[None,:],np.abs(O)**2*p[None,:]

def window(f,e):
 # Smooth positive-gap bandpass. No artificial lifetime broadening.
 return np.where(f>0.,f*f*np.exp(-.5*((f-e['center'])/e['width'])**2),0.)

def signal(lines,e):return float(np.sum(lines[1]*window(lines[0],e)))
def at(e,complete=True):return signal(gaussian_lines(e,complete),e)
def ref(e):return signal(fock_lines(e),e)

if __name__=='__main__':
 start=time.perf_counter();rng=np.random.default_rng(77371);candidate=[];reference_error=0.;cal_error=0.;db_error=0.;global_spin_error=0.;minimum=1.;normal_equiv=0.
 for j in range(100):
  e={'hopping':float(rng.uniform(.45,1.)), 'pairing':float(rng.uniform(.4,1.4)), 'temperature':float(rng.uniform(.2,.8)), 'site':int(rng.integers(3)), 'center':float(rng.uniform(.8,4.)), 'width':float(rng.uniform(.25,.65)), 'phase':float(rng.uniform(-1.5,1.5)), 'offset':float(rng.uniform(-.25,.25))}
  a=at(e);b=at(e,False);r=ref(e);reference_error=max(reference_error,abs(a-r));minimum=min(minimum,a,b)
  if a>.008 and abs(a-b)/a>.22 and abs(a-b)>.004:candidate.append({'input':e,'correct':a,'shortcut':b,'relative_error':abs(a-b)/a})
  zero=dict(e,pairing=0.);cal_error=max(cal_error,abs(at(zero)-at(zero,False)));normal_equiv=max(normal_equiv,abs(at(zero)-ref(zero)))
  f,w=gaussian_lines(e);# Gibbs detailed balance checked after rounding and grouping degenerate lines.
  buckets={}
  for x,z in zip(f.ravel(),w.ravel()):buckets[round(float(x),10)]=buckets.get(round(float(x),10),0.)+float(z)
  for x,z in buckets.items():db_error=max(db_error,abs(buckets.get(round(-x,10),0.)-np.exp(-x/e['temperature'])*z))
  global_spin_error=max(global_spin_error,abs(signal(gaussian_lines(e,True,[.5,-.5]*3),e)))
 candidates=sorted(candidate,key=lambda x:x['relative_error']);print('candidates',len(candidates),'ref',reference_error,'cal',cal_error,'normal-ref',normal_equiv,'DB',db_error,'globalspin',global_spin_error,'min',minimum,'seconds',time.perf_counter()-start)
 for x in candidates[:3]+candidates[-3:]:print(x)
 report={'status':'prototype_only_no_task_or_evaluation','candidate_count':len(candidate),'screened_cases':100,'max_gaussian_fock_absolute_error':reference_error,'calibration_shortcut_equality_error':cal_error,'normal_reference_error':normal_equiv,'detailed_balance_line_error':db_error,'global_spin_finite_frequency_error':global_spin_error,'minimum_screened_signal':minimum,'examples':candidates,'seconds':time.perf_counter()-start}
 Path('results/paired-spin-noise-prototype.json').write_text(json.dumps(report,indent=2)+'\n')
