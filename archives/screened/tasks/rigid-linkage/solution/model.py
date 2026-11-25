import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import i0e, i1e

ANGLES = np.linspace(-np.pi,np.pi,256,endpoint=False)
ANGLE_1 = ANGLES[:,None]
ANGLE_2 = ANGLES[None,:]
SIN_DIFFERENCE_SQUARED = np.sin(ANGLE_1-ANGLE_2)**2


class Model:
    def __init__(self):
        self.stiffness = None

    def fit(self, records):
        factors = np.array([r['input']['factors'][1] for r in records])
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def loss(stiffness):
            z = stiffness*factors
            prediction = i1e(z)/i0e(z)
            return np.sum(((prediction-values)/sigma)**2)
        result = minimize_scalar(loss,bounds=(.3,2.),method='bounded',options={'xatol':1e-12})
        self.stiffness = float(result.x)
        return self

    def predict(self, experiments):
        out = []
        for e in experiments:
            if e['locked']:
                first,second = 0.,ANGLES
                kinetic_weight = 1.
            else:
                first,second = ANGLE_1,ANGLE_2
                kinetic_weight = np.sqrt(e['mass_ratio']+SIN_DIFFERENCE_SQUARED)
            potential = self.stiffness*(e['factors'][0]*(1-np.cos(first-e['bias'][0]))
                                      +e['factors'][1]*(1-np.cos(second-e['bias'][1])))
            weight = kinetic_weight*np.exp(-potential)
            observable = np.cos(e['harmonic'][0]*first+e['harmonic'][1]*second+e['phase'])
            out.append(np.sum(weight*observable)/np.sum(weight))
        return np.array(out)
