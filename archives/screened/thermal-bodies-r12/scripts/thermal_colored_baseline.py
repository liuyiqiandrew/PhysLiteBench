from functools import lru_cache
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov
from numpy.polynomial.legendre import leggauss

NODES, WEIGHTS = leggauss(48)
NODES, WEIGHTS = (NODES+1)/2, WEIGHTS/2


def stiffness(springs, coupling):
    return np.diag(springs) + coupling*np.array([[1.,-1.],[-1.,1.]])


@lru_cache(512)
def stationary_state(friction,k1,k2,coupling,memory,mass,t1,t2):
    spring=stiffness([k1,k2],coupling)
    link=friction/memory
    # State order: main q, main p, coupler y, coupler P.
    drift=np.zeros((8,8))
    drift[:2,2:4]=np.eye(2)
    drift[2:4,:2]=-spring-link*np.eye(2)
    drift[2:4,4:6]=link*np.eye(2)
    drift[4:6,6:8]=np.eye(2)/mass
    drift[6:8,:2]=link*np.eye(2)
    drift[6:8,4:6]=-link*np.eye(2)
    drift[6:8,6:8]=-friction*np.eye(2)/mass
    diffusion=np.zeros((8,8))
    diffusion[6:8,6:8]=np.diag([2*friction*t1,2*friction*t2])
    covariance=solve_continuous_lyapunov(drift,-diffusion)
    return drift,covariance,np.linalg.eigh(spring)[1]


@lru_cache(2048)
def propagated(lag,friction,k1,k2,coupling,memory,mass,t1,t2):
    drift,covariance,_=stationary_state(friction,k1,k2,coupling,memory,mass,t1,t2)
    return expm(lag*drift)@covariance


def heat_variance(drift,covariance,friction,memory,mass,bath,duration,nodes=NODES,weights=WEIGHTS):
    if duration==0:return 0.
    link=friction/memory
    q,p,y,P=bath,2+bath,4+bath,6+bath
    power=np.zeros((8,8))
    power[p,y]=power[y,p]=link/2
    power[p,q]=power[q,p]=-link/2
    lag=duration*nodes
    evolution=np.array([expm(s*drift) for s in lag])
    work_kernel=evolution@(covariance@power@covariance)@evolution.transpose(0,2,1)
    work_correlation=2*np.einsum('ij,nji->n',power,work_kernel)
    value=2*duration*np.dot(weights,(duration-lag)*work_correlation)

    return float(value)


def predict_at(experiments,friction):
    out=[]
    for e in experiments:
        args=(float(friction),*e['springs'],e['coupling'],e['memory'],e['coupler_mass'],*e['temperatures'])
        drift,covariance,vectors=stationary_state(*args)
        if e['readout']=='mode_correlation':
            vector=vectors[:,e['mode']]
            temporal=propagated(e['lag'],*args)[2:4,2:4]
            value=vector@temporal@vector
        elif e['readout']=='heat_current':
            i=e['bath'];link=friction/e['memory']
            value=link*(covariance[2+i,4+i]-covariance[2+i,i])
        else:
            value=heat_variance(drift,covariance,friction,e['memory'],e['coupler_mass'],e['bath'],e['duration'])
        out.append(value)
    return np.asarray(out)


class Model:
    def __init__(self):self.friction=None

    def fit(self,records):
        from scipy.optimize import minimize_scalar
        inputs=[r['input'] for r in records]
        values=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        def objective(friction):
            residual=(predict_at(inputs,friction)-values)/sigma
            return float(residual@residual)
        self.friction=float(minimize_scalar(objective,bounds=(.3,1.2),method='bounded',options={'xatol':1e-12}).x)
        return self

    def predict(self,experiments):return predict_at(experiments,self.friction)
