from functools import lru_cache
import numpy as np
from scipy.linalg import eig, solve
from scipy.optimize import least_squares

TEMPERATURE=.65
Z=np.diag([1.,-1.])
X=np.array([[0.,1.],[1.,0.]])
LOWER=np.array([[0.,0.],[1.,0.]])
IDENTITY=np.eye(4)
X1=np.kron(X,np.eye(2));X2=np.kron(np.eye(2),X)
Z1=np.kron(Z,np.eye(2));Z2=np.kron(np.eye(2),Z)
OBSERVABLES=dict(excitation_1=(IDENTITY+Z1)/2,excitation_2=(IDENTITY+Z2)/2,xx_correlation=X1@X2)


def dissipator(jump):
    product=jump.conj().T@jump
    return np.kron(jump.conj(),jump)-.5*(np.kron(IDENTITY,product)+np.kron(product.T,IDENTITY))


def spectrum_factor(energy):
    return energy/(-np.expm1(-energy/TEMPERATURE))


@lru_cache(maxsize=64)
def setup(coupling):
    h=.5*Z1+.85*Z2+coupling*X1@X2
    energies,vectors=np.linalg.eigh(h)
    reservoirs=[]
    for gap,lower in [(1.,np.kron(LOWER,np.eye(2))),(1.7,np.kron(np.eye(2),LOWER))]:
        reservoirs.append(spectrum_factor(gap)*dissipator(lower)+spectrum_factor(-gap)*dissipator(lower.T))
    return h,energies,vectors,reservoirs


@lru_cache(maxsize=128)
def dynamics(coupling,rate):
    h,energies,vectors,reservoirs=setup(coupling)
    generator=-1j*(np.kron(IDENTITY,h)-np.kron(h.T,IDENTITY))+rate*sum(reservoirs)
    rates,modes=eig(generator)
    inverse=np.linalg.inv(modes)
    constrained=generator.copy()
    constrained[0]=np.eye(4).ravel(order='F')
    rhs=np.zeros(16);rhs[0]=1.
    rho_stationary=solve(constrained,rhs).reshape(4,4,order='F')
    return vectors,generator,rates,modes,inverse,rho_stationary



class Model:
    def __init__(self):
        self.rate=None

    def fit(self,records):
        raise NotImplementedError

    def predict(self,experiments):
        out=[]
        for e in experiments:
            if e['coupling']==0:
                if e['observable']=='xx_correlation':out.append(0.);continue
                spin=0 if e['observable']=='excitation_1' else 1
                gap=[1.,1.7][spin]
                equilibrium=1/(1+np.exp(gap/TEMPERATURE))
                initial=float(e['initial_level'] in ([1,3] if spin==0 else [2,3]))
                decay=0. if e['protocol']=='stationary' else np.exp(-self.rate*gap/np.tanh(gap/(2*TEMPERATURE))*e['time'])
                out.append(equilibrium+(initial-equilibrium)*decay)
                continue
            vectors,generator,rates,modes,inverse,rho_stationary=dynamics(e['coupling'],self.rate)
            if e['protocol']=='stationary':
                rho=rho_stationary
            else:
                state=vectors[:,e['initial_level']]
                initial=np.outer(state,state.conj()).ravel(order='F')
                evolved=modes@(np.exp(rates*e['time'])*(inverse@initial))
                rho=evolved.reshape(4,4,order='F')
            out.append(float(np.trace(rho@OBSERVABLES[e['observable']]).real))
        return np.array(out)
