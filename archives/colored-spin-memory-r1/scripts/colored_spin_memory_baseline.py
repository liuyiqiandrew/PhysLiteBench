import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

CORRELATION_TIME = 1.


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
        out = []
        for e in experiments:
            if all(np.linalg.norm(segment['field'])==0 for segment in e['segments']):
                time = sum(segment['duration'] for segment in e['segments'])
                vector = pulse(np.array([0.,0.,1.]),e['preparation'])
                memory = CORRELATION_TIME*(time-CORRELATION_TIME*(-np.expm1(-time/CORRELATION_TIME)))
                vector[:2] *= np.exp(-self.noise_width**2*memory)
                out.append((1+pulse(vector,e['readout'])[2])/2)
                continue
            vector = pulse(np.array([0.,0.,1.]),e['preparation'])
            elapsed = 0.
            for segment in e['segments']:
                duration = segment['duration']
                field = np.array(segment['field'])
                def rhs(t, state):
                    decay = self.noise_width**2*CORRELATION_TIME*(-np.expm1(-t/CORRELATION_TIME))
                    return np.cross(field,state)-decay*np.array([state[0],state[1],0.])
                if duration:
                    vector = solve_ivp(rhs,(elapsed,elapsed+duration),vector,rtol=2e-10,atol=2e-12,method='DOP853').y[:,-1]
                elapsed += duration
            vector = pulse(vector,e['readout'])
            out.append((1+vector[2])/2)
        return np.array(out)
