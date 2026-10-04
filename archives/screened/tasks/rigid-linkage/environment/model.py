import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import i0e, i1e

ANGLES = np.linspace(-np.pi,np.pi,256,endpoint=False)
ANGLE_1 = ANGLES[:,None]
ANGLE_2 = ANGLES[None,:]


class Model:
    def __init__(self):
        self.stiffness = None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        out = []
        for e in experiments:
            if e['locked']:
                first,second = 0.,ANGLES
            else:
                first,second = ANGLE_1,ANGLE_2
            potential = self.stiffness*(e['factors'][0]*(1-np.cos(first-e['bias'][0]))
                                      +e['factors'][1]*(1-np.cos(second-e['bias'][1])))
            weight = np.exp(-potential)
            observable = np.cos(e['harmonic'][0]*first+e['harmonic'][1]*second+e['phase'])
            out.append(np.sum(weight*observable)/np.sum(weight))
        return np.array(out)
