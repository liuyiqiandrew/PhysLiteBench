import numpy as np
from numpy.polynomial.legendre import leggauss
from functools import lru_cache
from scipy.optimize import minimize_scalar

DAMPING = .06
PRESSURE_SPEED = .1


@lru_cache(None)
def quadrature(order):
    return leggauss(order)


def mode_fields(z, experiment, plasma):
    omega = experiment['frequency']
    thickness = experiment['thickness']
    lateral = omega*np.sin(experiment['angle'])
    epsilon = 1-plasma**2/(omega*(omega+1j*DAMPING))
    conductivity = plasma**2/(DAMPING-1j*omega)
    transverse = np.sqrt(epsilon*omega**2-lateral**2+0j)
    longitudinal_squared = (omega*(omega+1j*DAMPING)-plasma**2)/PRESSURE_SPEED**2
    longitudinal = np.sqrt(longitudinal_squared-lateral**2+0j)
    fields = np.zeros((len(np.atleast_1d(z)),5,4),complex)
    for column,(wave_number,direction,anchor) in enumerate([
            (transverse,1,0),(transverse,-1,thickness),
            (longitudinal,1,0),(longitudinal,-1,thickness)]):
        phase = np.exp(1j*direction*wave_number*(np.atleast_1d(z)-anchor))
        if column<2:
            ex = direction*wave_number/(omega*epsilon)
            ez = -lateral/(omega*epsilon)
            state = np.array([ex,ez,1,conductivity*ex,conductivity*ez])
        else:
            state = np.array([-1j*lateral,-1j*direction*wave_number,0,
                              omega*lateral,omega*direction*wave_number])
        fields[:,:,column] = phase[:,None]*state
    return fields


def amplitudes(experiment, plasma):
    front,back = mode_fields([0,experiment['thickness']],experiment,plasma)
    cosine = np.cos(experiment['angle'])
    boundary = np.array([front[0]+cosine*front[2],back[0]-cosine*back[2],front[4],back[4]])
    return np.linalg.solve(boundary,np.array([2*cosine,0,0,0]))


def absorbed_fraction(experiment, plasma, order=64):
    nodes,weights = quadrature(order)
    low,high = np.array(experiment['window'])*experiment['thickness']
    z = (low+high)/2+(high-low)*nodes/2
    fields = mode_fields(z,experiment,plasma)@amplitudes(experiment,plasma)
    omega = experiment['frequency']
    loss = DAMPING/plasma**2*np.sum(abs(fields[:,3:5])**2,axis=1)
    return (high-low)*np.dot(weights,loss)/(2*np.cos(experiment['angle']))


def predict_at(experiments, plasma):
    cache={}
    values=[]
    for experiment in experiments:
        key=(experiment['frequency'],experiment['thickness'],experiment['angle'],tuple(experiment['window']))
        if key not in cache:cache[key]=absorbed_fraction(experiment,plasma)
        values.append(cache[key])
    return np.array(values)


class Model:
    def __init__(self):
        self.plasma_frequency = None

    def fit(self, records):
        inputs=[r['input'] for r in records]
        values=np.array([r['value'] for r in records])
        sigma=np.array([r['sigma'] for r in records])
        def objective(plasma):
            residual=(predict_at(inputs,plasma)-values)/sigma
            return residual@residual
        self.plasma_frequency=float(minimize_scalar(objective,bounds=(.85,1.15),method='bounded',options={'xatol':1e-12}).x)
        return self

    def predict(self, experiments):
        return predict_at(experiments,self.plasma_frequency)
