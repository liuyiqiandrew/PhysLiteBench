from functools import lru_cache
import numpy as np
from scipy.linalg import eigh
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

X=np.array([[0,1],[1,0]],complex)
Y=np.array([[0,-1j],[1j,0]])
Z=np.diag([1.,-1.])


@lru_cache(maxsize=512)
def phase(theta,q,r):
    def rhs(phi,flat):
        connection=(q*np.cos(r*phi)*X-q*np.cos(theta)*np.sin(r*phi)*Y+r*np.cos(theta)*Z)/2
        return (1j*connection@flat.reshape(2,2)).ravel()
    solution=solve_ivp(rhs,(0,2*np.pi),np.eye(2,dtype=complex).ravel(),method='DOP853',rtol=2e-11,atol=2e-13)
    transport=(-1)**(q+r)*solution.y[:,-1].reshape(2,2)
    return float(np.max(abs(np.angle(np.linalg.eigvals(transport))))/(2*np.pi))


@lru_cache(maxsize=512)
def levels(theta,q,r):
    n=np.arange(-24,25)
    shift=phase(theta,q,r)
    matrix=np.diag((n-shift)**2/2+np.sin(theta)**2*(r*r+q*q/2)/8)
    for i in range(len(n)-2*r):
        matrix[i,i+2*r]=matrix[i+2*r,i]=-np.sin(theta)**2*q*q/32
    return eigh(matrix,eigvals_only=True)


def predict_at(experiments,inertia):
    result=[]
    for e in experiments:
        energy=levels(e['theta'],e['q'],e['r'])/inertia
        weight=np.exp(-(energy-energy.min())/e['temperature'])
        result.append(float(weight@energy/weight.sum()))
    return np.array(result)


class Model:
    def __init__(self):
        self.inertia=None

    def fit(self,records):
        inputs=[r['input'] for r in records]
        values=np.array([r['value'] for r in records])
        sigma=np.array([r['sigma'] for r in records])
        def objective(inertia):
            residual=(predict_at(inputs,inertia)-values)/sigma
            return float(residual@residual)
        optimum=minimize_scalar(objective,bounds=(.8,1.2),method='bounded',options={'xatol':1e-13})
        self.inertia=float(min([.8,1.2,optimum.x],key=objective))
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.inertia)
