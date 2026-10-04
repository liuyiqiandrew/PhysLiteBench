import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar


def stress_readout(covariance, observable):
    if observable == 'shear_stress':
        return covariance[0,1]
    if observable == 'normal_stress':
        return covariance[0,0]-covariance[1,1]
    return np.trace(covariance)-2


def propagate(covariance, gradient, duration, relaxation_time):
    gradient = np.asarray(gradient,dtype=float)
    rotation = (gradient-gradient.T)/2
    drift = rotation-np.eye(2)/(2*relaxation_time)
    diffusion = np.eye(2)/relaxation_time+gradient+gradient.T
    generator = np.kron(drift,np.eye(2))+np.kron(np.eye(2),drift)
    augmented = np.zeros((5,5))
    augmented[:4,:4] = generator
    augmented[:4,4] = diffusion.reshape(-1)
    final = expm(augmented*duration)@np.r_[np.asarray(covariance).reshape(-1),1.]
    return final[:4].reshape(2,2)


class Model:
    def __init__(self):
        self.relaxation_time = None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        out = []
        for e in experiments:
            covariance = np.array(e['initial_covariance'],dtype=float)
            for s in e['segments']:
                covariance = propagate(covariance,s['gradient'],s['duration'],self.relaxation_time)
            out.append(stress_readout(covariance,e['observable']))
        return np.array(out)
