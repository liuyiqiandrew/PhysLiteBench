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
        times = np.array([sum(s['duration'] for s in r['input']['segments']) for r in records])
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        memory = CORRELATION_TIME*(times-CORRELATION_TIME*(-np.expm1(-times/CORRELATION_TIME)))
        def loss(width):
            prediction = .5*(1+np.exp(-width**2*memory))
            return np.sum(((prediction-values)/sigma)**2)
        fit = minimize_scalar(loss,bounds=(.3,1.6),method='bounded',options={'xatol':1e-12})
        self.noise_width = float(fit.x)
        return self

    def predict(self, experiments):
        n = MODES
        diagonal = sparse.diags(-np.arange(n)/CORRELATION_TIME)
        neighbors = sparse.diags([np.sqrt(np.arange(1,n)),np.sqrt(np.arange(1,n))],[-1,1],shape=(n,n))
        noise = sparse.kron(diagonal,np.eye(3))+self.noise_width*sparse.kron(neighbors,cross_matrix([0.,0.,1.]))
        out = []
        for e in experiments:
            if all(np.linalg.norm(segment['field'])==0 for segment in e['segments']):
                time = sum(segment['duration'] for segment in e['segments'])
                vector = pulse(np.array([0.,0.,1.]),e['preparation'])
                memory = CORRELATION_TIME*(time-CORRELATION_TIME*(-np.expm1(-time/CORRELATION_TIME)))
                vector[:2] *= np.exp(-self.noise_width**2*memory)
                out.append((1+pulse(vector,e['readout'])[2])/2)
                continue
            state = np.zeros(3*n)
            state[:3] = pulse(np.array([0.,0.,1.]),e['preparation'])
            for segment in e['segments']:
                generator = noise+sparse.kron(sparse.eye(n),cross_matrix(segment['field']))
                state = expm_multiply(generator*segment['duration'],state)
            vector = pulse(state[:3],e['readout'])
            out.append((1+vector[2])/2)
        return np.array(out)
