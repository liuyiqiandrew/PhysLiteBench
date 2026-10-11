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
        raise NotImplementedError('Fit the reference electrostatic coupling.')

    def predict(self,experiments):
        return predict_at(experiments,self.coupling)
