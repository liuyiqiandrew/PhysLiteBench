import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.linalg import solve_continuous_lyapunov

def covariance(m,gamma,K,T,B):
    J=np.array([[0.,1.],[-1.,0.]])
    A=np.block([[np.zeros((2,2)),np.sqrt(m)*np.eye(2)],[-np.sqrt(m)*K,-gamma*np.eye(2)+B*J]])
    Q=np.zeros((4,4));Q[2:,2:]=2*gamma*np.diag(T)
    C=solve_continuous_lyapunov(A,-Q);C=(C+C.T)/2
    heat=gamma*(T-np.diag(C)[2:])/m
    Y=C[:2,2:]/np.sqrt(m);V=C[2:,2:]/m
    trap=np.diag(K@Y);magnetic=-B*np.array([V[0,1],-V[0,1]])
    return C,heat,trap,magnetic,float(np.max(abs(A@C+C@A.T+Q)))

def spectral(m,gamma,K,T,B):
    h=K[0,1]
    def f(w):
        determinant=(K[0,0]-m*w*w-1j*gamma*w)*(K[1,1]-m*w*w-1j*gamma*w)-h*h-B*B*w*w
        return w*w*(h*h+B*B*w*w)/abs(determinant)**2
    integral,error=quad(f,0,np.inf,epsabs=2e-11,epsrel=2e-11,limit=200)
    return 2*gamma*gamma*(T[0]-T[1])*integral/np.pi
rng=np.random.default_rng(673)
checks=[]
for _ in range(48):
    gamma=rng.uniform(.7,1.5); k1,k2=rng.uniform(.8,1.8,2); h=rng.uniform(-.55,.55)
    K=np.array([[k1,h],[h,k2]]);T=rng.uniform(.5,2.,2);m=rng.uniform(.15,.8); B=rng.uniform(-1.2,1.2)
    C,heat,trap,mag,res=covariance(m,gamma,K,T,B)
    ref=spectral(m,gamma,K,T,B)
    checks.append(dict(gamma=gamma,mass=m,K=K.tolist(),T=T.tolist(),B=B,oracle=float(heat[0]),reference=ref,error=abs(heat[0]-ref),trap=float(trap[0])))
r=dict(status='finite_mass_prototype_only',max_reference_error=max(x['error'] for x in checks),checks=checks)
Path(__file__).with_name('frequency-report.json').write_text(json.dumps(r,indent=2)+'\n')
print('spectral_error',r['max_reference_error'])
