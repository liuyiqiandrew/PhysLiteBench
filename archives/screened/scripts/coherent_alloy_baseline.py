import numpy as np
from scipy.optimize import minimize_scalar

LAME_LAMBDA=2.
SHEAR_MODULUS=3.
EIGENSTRAIN=np.diag([0.,.5,-.5])
CHEMICAL_CURVATURE=.4
GRADIENT_ENERGY=.2


def elastic_penalty(direction):
    eigenstress=LAME_LAMBDA*np.trace(EIGENSTRAIN)*np.eye(3)+2*SHEAR_MODULUS*EIGENSTRAIN
    penalty=float(np.sum(EIGENSTRAIN*eigenstress))

    return penalty


def decay_rates(experiments):
    values=[]
    for e in experiments:
        q=np.asarray(e['wavevector'],dtype=float);square=float(q@q)
        penalty=elastic_penalty(q/np.sqrt(square))
        values.append(square*(CHEMICAL_CURVATURE+GRADIENT_ENERGY*square+penalty))
    return np.array(values)


class Model:
    def __init__(self):
        self.mobility=None

    def fit(self,records):
        inputs=[r['input'] for r in records];rate=decay_rates(inputs)
        time=np.array([e['time'] for e in inputs]);amplitude=np.array([e['amplitude'] for e in inputs])
        value=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        self.mobility=float(minimize_scalar(lambda m: np.sum(((amplitude*np.exp(-m*rate*time)-value)/sigma)**2),bounds=(.03,.2),method='bounded',options={'xatol':1e-13}).x)
        return self

    def predict(self,experiments):
        amplitude=np.array([e['amplitude'] for e in experiments]);time=np.array([e['time'] for e in experiments])
        return amplitude*np.exp(-self.mobility*decay_rates(experiments)*time)
