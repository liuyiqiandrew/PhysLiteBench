import numpy as np
from numpy.polynomial.legendre import leggauss


def predict_at(experiments, coupling, order=80):
    if not experiments:
        return np.empty(0)
    nodes, weights = leggauss(order)
    bands = np.array([e['band'] for e in experiments])
    half_width = (bands[:,1]-bands[:,0])/2
    omega = bands.mean(axis=1)[:,None] + half_width[:,None]*nodes
    shift = np.array([e['shift'] for e in experiments])[:,None]
    left = 1/np.expm1(omega/np.array([e['left_temperature'] for e in experiments])[:,None])
    right = 1/np.expm1(omega/np.array([e['right_temperature'] for e in experiments])[:,None])
    denominator = (.35+1j*(1.4+shift-omega))*(.55+1j*(1.9+shift-omega))+coupling**2
    transmission = .7*1.1*coupling**2/abs(denominator)**2
    current = omega*transmission*(left-right)
    forward_rate = transmission*left*(1+right)
    backward_rate = transmission*right*(1+left)
    variance = omega**2*(forward_rate+backward_rate)
    integrands = np.where(np.array([e['readout']=='current' for e in experiments])[:,None], current, variance)
    return half_width*np.sum(integrands*weights,axis=1)/(2*np.pi)


class Model:
    def __init__(self):
        self.coupling = None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        return predict_at(experiments,self.coupling)
