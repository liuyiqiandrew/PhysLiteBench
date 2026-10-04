import numpy as np


def response(experiment):
    a=experiment['acceleration'];gap=experiment['gap']
    c=0.
    if experiment['trajectory']=='circle':
        v=experiment['speed'];gamma=1/np.sqrt(1-v*v)
        c=a*a/(30*(gamma*v)**2)
    scale=np.sqrt(c+a*a/12)
    excitation=a*a/(48*np.pi*scale)*np.exp(-gap/scale)
    return float(excitation+(gap/(2*np.pi) if experiment['readout']=='deexcitation' else 0.))


class Model:
    def __init__(self):self.coupling=None

    def fit(self,records):
        basis=np.array([response(r['input']) for r in records])
        y=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        self.coupling=float(np.sum(basis*y/sigma**2)/np.sum(basis*basis/sigma**2))
        return self

    def predict(self,experiments):return self.coupling*np.array([response(e) for e in experiments])
