import json,time
import numpy as np
from scipy.linalg import expm,solve_continuous_lyapunov

def cell(a,ap,theta,thetap,alpha,D):
 c=np.cos(theta);s=np.sin(theta);R=np.array([[c,-s],[s,c]]);J=np.array([[0,-1],[1,0]])
 B=a*R;Bp=ap*R+a*thetap*J@R
 A=np.block([[np.eye(2),-B],[np.zeros((2,2)),np.eye(2)/alpha]])
 Ap=np.block([[np.zeros((2,2)),-Bp],[np.zeros((2,2)),np.zeros((2,2))]])
 Q=np.diag([0,0,2*D/alpha**2,2*D/alpha**2]);C=solve_continuous_lyapunov(A,Q)
 P=np.linalg.solve(A.T,np.c_[np.eye(2),np.zeros((2,2))].T).T
 Pp=np.linalg.solve(A.T,(-P@Ap).T).T
 diffusion=C[:2]@P.T
 return Pp@C[0],(diffusion+diffusion.T)/2

def mc(m,alpha=1.2,D=.8,k=2,dt=.08,steps=10000,n=2048,seed=174):
 # Fast-time exact OU substep, split by physical rotation of the actuator force.
 # Frame coordinates permit translation-invariant stationary simulation at a=1,F=0.
 A=np.block([[-np.eye(2),np.eye(2)],[np.zeros((2,2)),-np.eye(2)/alpha]])
 Q=np.diag([0,0,2*D/alpha**2,2*D/alpha**2]);C=solve_continuous_lyapunov(A,-Q);E=expm(A*dt)
 noise=C-E@C@E.T;L=np.linalg.cholesky(noise)
 rng=np.random.default_rng(seed);state=rng.multivariate_normal(np.zeros(4),C,n)
 sums=np.zeros(n);samples=0
 for i in range(steps+1000):
  angle=.5*dt*k*np.sqrt(m)*state[:,0];c=np.cos(angle);s=np.sin(angle);fx=state[:,2].copy();fy=state[:,3].copy();state[:,2]=c*fx-s*fy;state[:,3]=s*fx+c*fy
  state=state@E.T+rng.normal(size=(n,4))@L.T
  angle=.5*dt*k*np.sqrt(m)*state[:,0];c=np.cos(angle);s=np.sin(angle);fx=state[:,2].copy();fy=state[:,3].copy();state[:,2]=c*fx-s*fy;state[:,3]=s*fx+c*fy
  if i>=1000:sums+=alpha*k*state[:,0]**2;samples+=1
 averages=sums/samples
 return dict(mass=m,dt=dt,mean=float(averages.mean()),standard_error=float(averages.std(ddof=1)/np.sqrt(n)),expected_limit=alpha*k*D/(1+alpha))
if __name__ == "__main__":
    start=time.time();r={'cell_spots':[]}
    for alpha in [.25,1.2,4]:
     b,d=cell(1.3,.4,.7,-1.2,alpha,.8);expected=alpha/(1+alpha)*.8*np.array([1.3*.4,1.3**2*(-1.2)])
     r['cell_spots'].append({'alpha':alpha,'drift':b.tolist(),'expected':expected.tolist(),'diffusion':d.tolist()})
    r['finite_mass']=[mc(m) for m in [.02,.005,.00125]]
    r['time_refinement']=mc(.00125,dt=.04,steps=20000)
    r['small_mass_refinement']=mc(.0003125,dt=.04,steps=20000,n=8192,seed=77513)
    r['seconds']=time.time()-start
    print(json.dumps(r,indent=2));open('results/actuator-frame-current-prototype.json','w').write(json.dumps(r,indent=2)+'\n')
