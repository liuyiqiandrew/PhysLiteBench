import numpy as np
from scipy import sparse
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import expm_multiply

CORRELATION_TIME = 1.
MODES = 40


def cross_matrix(vector):
    x,y,z = vector
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def pulse(vector, specification):
    phase, angle = specification
    axis = np.array([np.cos(phase),np.sin(phase),0.])
    return vector*np.cos(angle)+np.cross(axis,vector)*np.sin(angle)+axis*np.dot(axis,vector)*(1-np.cos(angle))


class Model:
    def __init__(self):
        self.noise_width = None

    def fit(self, records):
        raise NotImplementedError

    def _segment_channel(self, field, duration):
        """Return the three-dimensional Bloch map for a constant-control interval."""
        if np.linalg.norm(field) == 0:
            memory = CORRELATION_TIME*(duration-CORRELATION_TIME*(-np.expm1(-duration/CORRELATION_TIME)))
            coherence = np.exp(-self.noise_width**2*memory)
            return np.diag([coherence,coherence,1.])
        n = MODES
        diagonal = sparse.diags(-np.arange(n)/CORRELATION_TIME)
        neighbors = sparse.diags([np.sqrt(np.arange(1,n)),np.sqrt(np.arange(1,n))],[-1,1],shape=(n,n))
        generator = (sparse.kron(diagonal,np.eye(3))
                     +sparse.kron(sparse.eye(n),cross_matrix(field))
                     +self.noise_width*sparse.kron(neighbors,cross_matrix([0.,0.,1.])))
        initial = np.zeros((3*n,3))
        initial[:3,:] = np.eye(3)
        propagated = expm_multiply(generator*duration,initial)
        return propagated[:3,:]

    def predict(self, experiments):
        out = []
        for e in experiments:
            vector = pulse(np.array([0.,0.,1.]),e['preparation'])
            for segment in e['segments']:
                channel = self._segment_channel(segment['field'],segment['duration'])
                vector = channel@vector
            vector = pulse(vector,e['readout'])
            out.append((1+vector[2])/2)
        return np.array(out)
