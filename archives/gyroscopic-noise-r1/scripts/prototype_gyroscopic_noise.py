import json
from pathlib import Path
import numpy as np
from scipy.linalg import solve_continuous_lyapunov
J=np.array([[0.,1.],[-1.,0.]])
def matrices(omega,gamma,field,angle,stiffness=(1.,2.1)):
 r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
 k=r@np.diag(stiffness)@r.T
 chi=np.linalg.inv(k-omega**2*np.eye(2)-1j*omega*(gamma*np.eye(2)-field*J))
 s=2*gamma*chi@chi.conj().T
 approx=2/omega*(chi.imag+chi.imag.T)/2
 drift=np.block([[np.zeros((2,2)),np.eye(2)],[-k,-gamma*np.eye(2)+field*J]])
 diffusion=np.diag([0.,0.,2*gamma,2*gamma])
 cov=solve_continuous_lyapunov(drift,-diffusion)
 resolvent=np.linalg.inv(-1j*omega*np.eye(4)-drift)
 ref=(resolvent@diffusion@resolvent.conj().T)[:2,:2]
 return s,approx,ref,cov,k
cases=[]
for field,omega,angle,tau,gain in [(1.2,.7,.2,1.8,.8),(.8,1.1,.6,1.4,1.),(-1.3,.9,-.3,2.,.7),(1.6,1.4,.8,.8,1.2),(1.3,.5,.4,2.5,1.)]:
 s,a,r,cov,k=matrices(omega,.65,field,angle)
 h=np.array([1.,gain*np.exp(1j*omega*tau)])
 true=(h@s@h.conj()).real
 short=(h@a@h.conj()).real
 h0=np.array([1.,gain])
 cases.append({'field':field,'omega':omega,'delay':tau,'oracle':true,'shortcut':short,'relative_error':abs(short/true-1),'cal_error':abs(h0@(s-a)@h0),'reference_error':float(abs(s-r).max()),'shortcut_min_eigen':float(np.linalg.eigvalsh(a).min()),'gibbs_error':float(abs(cov-np.block([[np.linalg.inv(k),np.zeros((2,2))],[np.zeros((2,2)),np.eye(2)]])).max())})
report = [{key: float(value) for key, value in case.items()} for case in cases]
path = Path(__file__).resolve().parents[1]/'results/gyroscopic-noise-prototype.json'
path.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
