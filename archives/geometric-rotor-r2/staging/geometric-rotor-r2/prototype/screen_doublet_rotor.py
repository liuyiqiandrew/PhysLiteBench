import importlib.util,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
s=importlib.util.spec_from_file_location('p','/private/tmp/prototype_doublet_rotor.py');p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
def scalar(theta,T,I=1.07,q=1,r=1,physical=True,M=24):
 a=p.holonomy(theta,q,r)[0] if physical else .5*((q+r)%2)
 n=np.arange(-M,M+1);H=np.diag((n-a)**2/(2*I)+np.sin(theta)**2*(r*r+q*q/2)/(8*I))
 for i in range(len(n)-2*r):H[i,i+2*r]=H[i+2*r,i]=-np.sin(theta)**2*q*q/(32*I)
 E=eigh(H,eigvals_only=True);w=np.exp(-(E-E[0])/T);return float(w@E/w.sum())
start=time.perf_counter();examples=[]
for q,r in [(1,2),(2,1),(2,3),(3,2)]:
 for theta in [.3,.5,.7,.9,1.1,1.3,np.pi/2]:
  for T in [.04,.1,.2]:
   a=scalar(theta,T,q=q,r=r);b=scalar(theta,T,q=q,r=r,physical=False)
   examples.append(dict(q=q,r=r,tilt=theta,temperature=T,correct=a,shortcut=b,relative_gap=abs(a-b)/a,holonomy_phase=p.holonomy(theta,q,r)[0]))
checks=[]
for e in [dict(q=2,r=1,tilt=.5,temperature=.1),dict(q=2,r=1,tilt=.7,temperature=.1),dict(q=1,r=2,tilt=.5,temperature=.1)]:
 vals=[p.finite(e['tilt'],e['temperature'],q=e['q'],r=e['r'],gap=d) for d in [1024,2048,4096]]
 exact=scalar(e['tilt'],e['temperature'],q=e['q'],r=e['r']);ref=(vals[0]-6*vals[1]+8*vals[2])/3
 checks.append(dict(input=e,oracle=exact,finite_gap=ref,error=abs(exact-ref)))
report={'status':'prototype_only_no_task_or_evaluation','source':'Exact local scalar geometric potential plus correct periodic/antiperiodic frame endpoint; drops only path-ordered doublet transport.','examples':examples,'finite_gap_checks':checks,'seconds':time.perf_counter()-start,'max_calibration_error':max(abs(e['correct']-e['shortcut']) for e in examples if e['tilt']==np.pi/2),'physical_note':'Arbitrary integer q,r leave the lab Hamiltonian periodic, but U(2pi)=(-1)^(q+r)I. Calibration tilt pi/2 has identity path-ordered connection transport; total lab monodromy is the central parity sign, already retained by source.'}
Path('results/doublet-rotor-endpoint-prototype.json').write_text(json.dumps(report,indent=2)+'\n')
print('MAXCAL',report['max_calibration_error'],'REF',checks,'seconds',report['seconds'])
for e in examples:
 if e['q']==2 and e['r']==1:print(e)
