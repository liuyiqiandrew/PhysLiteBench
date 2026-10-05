from functools import lru_cache
import numpy as np


@lru_cache(None)
def modes(coupling):
    derivative = np.eye(4, 5, k=1)-np.eye(4, 5)
    matrix = .6*np.eye(5)+coupling*derivative.T@derivative
    return np.linalg.eigh(matrix)


@lru_cache(maxsize=16384)
def response(stiffness, temperature, coupling, wavevector, delay):
    values, vectors = modes(coupling)
    variance = temperature/(stiffness*values)
    frequency = np.sqrt(stiffness*values-.04)
    decay = np.exp(-.2*delay)*(np.cos(frequency*delay)
                              +.2*np.sin(frequency*delay)/frequency)
    covariance = (vectors*variance)@vectors.T
    diagonal = np.diag(covariance)
    phase = np.exp(1j*wavevector*np.arange(5))
    weights = np.real(phase[:,None]*phase.conj()[None,:])/5
    pair_variance = diagonal[:,None]+diagonal[None,:]
    time_covariance = (vectors*(variance*decay))@vectors.T
    return float(np.sum(weights*np.exp(-wavevector**2*(pair_variance-2*time_covariance)/2)))


def predict_at(experiments, stiffness):
    return np.asarray([response(float(stiffness),float(e['temperature']),float(e['coupling']),
                                float(e['wavevector']),float(e['delay'])) for e in experiments])


class Model:
    def __init__(self):
        self.stiffness = None

    def fit(self, records):
        from scipy.optimize import minimize_scalar
        inputs = [r['input'] for r in records]
        values = np.asarray([r['value'] for r in records])
        sigma = np.asarray([r['sigma'] for r in records])
        def loss(k):
            return float(np.sum(((predict_at(inputs,k)-values)/sigma)**2))
        result = minimize_scalar(loss,bounds=(.8,1.2),method='bounded',options={'xatol':1e-12})
        self.stiffness = min([.8,float(result.x),1.2],key=loss)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.stiffness)
