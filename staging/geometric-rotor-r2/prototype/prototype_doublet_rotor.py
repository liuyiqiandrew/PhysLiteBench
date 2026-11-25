"""Prototype: rank-two adiabatic rotor and SU(2) boundary holonomy."""
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
import json,time
s0=np.eye(2);sx=np.array([[0,1],[1,0]],complex);sy=np.array([[0,-1j],[1j,0]]);sz=np.diag([1.,-1.]);P0=np.kron((s0+sz)/2,s0)

def matrices(phi,theta,q=1,r=1):
 G=np.kron(s0,sx);J=np.kron(np.cos(theta)*sz+np.sin(theta)*sx,sz)
 U1=np.cos(q*phi/2)*np.eye(4)-1j*np.sin(q*phi/2)*G
 U2=np.cos(r*phi/2)*np.eye(4)-1j*np.sin(r*phi/2)*J
 U=U1@U2
 dU=(-1j*q/2*G)@U+U1@(-1j*r/2*J)@U2
 return U,dU

def connection(phi,theta,q=1,r=1):return .5*(q*np.cos(r*phi)*sx-q*np.cos(theta)*np.sin(r*phi)*sy+r*np.cos(theta)*sz)

def holonomy(theta,q=1,r=1):
 def f(t,y):return (1j*connection(t,theta,q,r)@y.reshape(2,2)).ravel()
 sol=solve_ivp(f,(0,2*np.pi),np.eye(2,dtype=complex).ravel(),rtol=2e-11,atol=2e-13,method='DOP853')
 W=sol.y[:,-1].reshape(2,2)
 # Lab U is periodic when q+r is even, otherwise includes a minus sign.
 W*=(-1)**(q+r)
 phase=float(np.max(np.abs(np.angle(np.linalg.eigvals(W))))/(2*np.pi))
 return phase,W

def scalar(theta,T,I=1.07,q=1,r=1,physical=True,M=24):
 a=holonomy(theta,q,r)[0] if physical else 0.
 n=np.arange(-M,M+1);H=np.diag((n-a)**2/(2*I)+np.sin(theta)**2*(r*r+q*q/2)/(8*I))
 for i in range(len(n)-2*r):H[i,i+2*r]=H[i+2*r,i]=-np.sin(theta)**2*q*q/(32*I)
 E=eigh(H,eigvals_only=True);w=np.exp(-(E-E[0])/T);return float(w@E/w.sum())

def finite(theta,T,I=1.07,q=1,r=1,gap=1024,M=14):
 n=np.arange(-M,M+1);ph=2*np.pi*np.arange(64)/64
 potential=[]
 for x in ph:
  U,_=matrices(x,theta,q,r);potential.append(np.eye(4)-U@P0@U.conj().T)
 coeff=np.fft.fft(potential,axis=0)/len(ph)
 K=np.repeat(n*n/(2*I),4);H=np.diag(K).astype(complex)
 for i,ni in enumerate(n):
  for j,nj in enumerate(n):H[4*i:4*i+4,4*j:4*j+4]+=gap*coeff[(ni-nj)%len(ph)]
 E,V=eigh(H);p=np.exp(-(E-E[0])/T);p/=p.sum();return float(p@(abs(V)**2).T@K)

if __name__=='__main__':
 start=time.perf_counter();cases=[];referr=0.;refine=0.;sourcecal=0.;matrixerror=0.;bherror=0.;comm=0.;trivial=0.
 for q,r in [(1,1),(2,2),(3,1)]:
  for theta in [.25,.55,.85,1.15,np.pi/2]:
   for T in [.04,.12,.3]:
    exact=scalar(theta,T,q=q,r=r);wrong=scalar(theta,T,q=q,r=r,physical=False)
    vals=[finite(theta,T,q=q,r=r,gap=d) for d in [512,1024,2048]]
    extrap=(vals[0]-6*vals[1]+8*vals[2])/3
    error=abs(exact-extrap);referr=max(referr,error)
    a,W=holonomy(theta,q,r)
    if theta==np.pi/2:sourcecal=max(sourcecal,abs(exact-wrong));trivial=max(trivial,np.linalg.norm(W-np.eye(2)))
    cases.append(dict(q=q,r=r,tilt=theta,temperature=T,holonomy_phase=a,correct=exact,shortcut=wrong,relative_gap=abs(exact-wrong)/exact,finite_gap_extrapolated=extrap,reference_error=error))
  for x in [.2,1.1,2.7]:
   U,dU=matrices(x,.8,q,r);A=1j*(U.conj().T@dU)[:2,:2];Phi=(dU.conj().T@(np.eye(4)-U@P0@U.conj().T)@dU)[:2,:2]/(2*1.07)
   matrixerror=max(matrixerror,np.max(abs(A-connection(x,.8,q,r))));bherror=max(bherror,np.max(abs(Phi-np.eye(2)*np.sin(.8)**2*(q*q*np.sin(r*x)**2+r*r)/(8*1.07))))
  A=connection(.2,.8,q,r);B=connection(1.2,.8,q,r);comm=max(comm,np.linalg.norm(A@B-B@A))
 for theta,T,q,r in [(.55,.04,1,1),(.85,.12,2,2),(1.15,.3,3,1)]:
  a=(finite(theta,T,q=q,r=r,gap=512)-6*finite(theta,T,q=q,r=r,gap=1024)+8*finite(theta,T,q=q,r=r,gap=2048))/3
  b=(finite(theta,T,q=q,r=r,gap=1024)-6*finite(theta,T,q=q,r=r,gap=2048)+8*finite(theta,T,q=q,r=r,gap=4096))/3
  refine=max(refine,abs(a-b))
 report={'status':'prototype_only_no_task_or_evaluation','source_calibration_difference':sourcecal,'trivial_calibration_holonomy_error':trivial,'connection_from_full_frame_error':float(matrixerror),'geometric_scalar_from_full_frame_error':float(bherror),'connection_noncommutator_norm':float(comm),'finite_gap_reference_max_error':referr,'gap_refinement_error':refine,'cases':cases,'seconds':time.perf_counter()-start}
 Path('results/doublet-rotor-prototype.json').write_text(json.dumps(report,indent=2)+'\n')
 print({k:v for k,v in report.items() if k!='cases'})
 for x in cases:
  if x['relative_gap']>.12:print(x)
