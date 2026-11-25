import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar


def coefficient(gamma,count,ratio):
    length = np.log(ratio)
    slopes = 2-gamma*np.arange(count,0,-1)
    sums = np.r_[0,np.cumsum(slopes)]
    generator = np.diag(-sums)+np.diag(np.ones(count),-1)
    initial = np.zeros(count+1)
    initial[0] = 1.
    state = expm(length*generator)@initial
    derivative = sums[-1]+(generator@state)[-1]/state[-1]
    return -np.expm1(-2*length)*derivative/(2*count)


def predict_at(experiments,coupling):
    return np.asarray([coefficient(e['temperature_multiplier']*coupling,
                                  e['ion_count'],e['radius_ratio']) for e in experiments])


class Model:
    def __init__(self):
        self.coupling = None

    def fit(self,records):
        inputs = [r['input'] for r in records]
        values = np.asarray([r['value'] for r in records])
        sigma = np.asarray([r['sigma'] for r in records])
        def objective(coupling):
            residual = (predict_at(inputs,coupling)-values)/sigma
            return float(residual@residual)
        result = minimize_scalar(objective,bounds=(.7,1.4),method='bounded',
                                 options={'xatol':1e-12})
        self.coupling = float(result.x)
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.coupling)
