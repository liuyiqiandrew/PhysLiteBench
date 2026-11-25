import json
import numpy as np
from scipy.linalg import eigh
from itertools import product
from pathlib import Path
Sx=np.array([[0,1,0],[1,0,1],[0,1,0]],float)/np.sqrt(2)
Sz=np.diag([1.,0.,-1.]); Q=Sz@Sz

def state(D,hx,hz,T,shift=0):
 H=(D-shift)*Q+hx*Sx+hz*Sz
 E,V=eigh(H); p=np.exp(-(E-E[0])/T); p/=p.sum(); A=V.T@Q@V
 return E,p,A

def responses(D,hx,hz,T):
 E,p,A=state(D,hx,hz,T); mean=np.dot(p,np.diag(A))
 var=np.dot(p,np.diag(A@A))-mean**2
 chi=np.dot(p,np.diag(A)**2)/T-mean**2/T
 for i in range(3):
  for j in range(i+1,3):
   d=E[j]-E[i]
   chi+=2*(p[i]-p[j])/d*A[i,j]**2 if d>1e-10 else 2*p[i]/T*A[i,j]**2
 def force(shift):
  e,w,a=state(D,hx,hz,T,shift)
  return np.dot(w,np.diag(a))
 eps=1e-3
 slope=lambda h:(-force(2*h)+8*force(h)-8*force(-h)+force(-2*h))/(12*h)
 ref=(16*slope(eps/2)-slope(eps))/15
 return dict(exact=chi,source=var/T,reference=ref,relative_excess=var/(T*chi)-1)

cal=[responses(*x) for x in product([.7,1,1.3],[0],[0,.3,.7],[.25,.5,.8])]
cases=[]
for x in product([.7,1.,1.3],[.7,1.1,1.5],[0,.35,.7],[.2,.35,.5]):
 d=responses(*x); d['controls']=x; cases.append(d)
healthy=[x for x in cases if x['relative_excess']>.25 and x['exact']>.08]
report={'apparatus':'Spin-1 molecule, H=(D-lambda*strain)*Sz^2+hx*Sx+hz*Sz; each imposed strain is fully Gibbs equilibrated. Read out slope of conjugate molecular force versus strain. Responses tabulated per lambda^2.',
 'calibration_max_gap':max(abs(x['exact']-x['source']) for x in cal),
 'reference_max_absolute_error':max(abs(x['reference']-x['exact']) for x in cases),
 'source_relative_excess_range':[min(x['relative_excess'] for x in cases),max(x['relative_excess'] for x in cases)],
 'exact_signal_range':[min(x['exact'] for x in cases),max(x['exact'] for x in cases)],
 'healthy_count':len(healthy),'case_count':len(cases),'examples':healthy[::max(1,len(healthy)//6)][:6],
 'parameter_identifiability':'Positive coupling lambda enters exact calibration as lambda^2 times a strictly positive known variance/T, so unique positive WLS recovery in any bounded positive domain.',
 'primary_source':'https://journals.aps.org/prb/abstract/10.1103/PhysRevB.94.075121',
 'scope':'Prototype only; no task package, noise generation, grader or model trials.'}
Path('/private/tmp/quantum_compliance_prototype.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
