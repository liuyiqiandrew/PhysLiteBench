import numpy as np,json
from scipy.linalg import eigh
A=np.array([[1.8,.21,-.13],[.21,2.2,.17],[-.13,.17,2.6]])
B=np.array([[1.2,.1,.18],[.1,1.6,-.14],[.18,-.14,1.7]])
def solve(squeeze,temperature,scale=1.04):
 n=len(A);b=squeeze*B;D=np.block([[A,b],[-b,-A]]);vals,vec=np.linalg.eig(D);ix=np.argsort(vals.real)[n:];energies=vals[ix].real;u=vec[:,ix].real
 metric=np.r_[np.ones(n),-np.ones(n)];norm=np.sum(metric[:,None]*u*u,axis=0)
 weights=u[:n]**2+u[n:]**2;pop=1/np.expm1(scale*energies/temperature)
 shortcut=weights@pop;good=(weights/norm)@pop
 G=np.block([[A+b,np.zeros((n,n))],[np.zeros((n,n)),A-b]])
 e,v=eigh(G);root=(v*np.sqrt(e))@v.T;inv=(v/np.sqrt(e))@v.T;J=np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]])
 F=root@J@root;z,U=eigh(-F@F);omega=np.sqrt(z);dv=inv@(U*(omega/np.expm1(scale*omega/temperature)))@U.T@inv
 ref=.5*(np.diag(dv)[:n]+np.diag(dv)[n:])
 return dict(squeeze=squeeze,T=temperature,shortcut=shortcut.tolist(),oracle=good.tolist(),relative_error=(abs(shortcut/good-1)).tolist(),oracle_ref=max(abs(good-ref)),mode_norms=norm.tolist(),min_G_eigen=e.min())
print(json.dumps([solve(s,t) for s in [.5,.8,1.] for t in [.3,.7,1.2]],indent=2))
