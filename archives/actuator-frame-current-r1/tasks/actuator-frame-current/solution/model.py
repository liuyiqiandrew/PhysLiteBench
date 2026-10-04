import numpy as np
from functools import lru_cache
from scipy.optimize import minimize_scalar


@lru_cache(None)
def geometry(contrast, winding, twist, order=512):
    x = 2*np.pi*np.arange(order)/order
    a = 1+contrast*np.cos(x)
    derivative = -contrast*np.sin(x)
    theta = winding*x+twist*np.sin(x)
    theta_derivative = winding+twist*np.cos(x)
    c,s = np.cos(theta),np.sin(theta)
    rotation = np.array([[c,-s],[s,c]]).transpose(2,0,1)
    angular_derivative = np.array([[-s,-c],[c,-s]]).transpose(2,0,1)
    coupling = a[:,None,None]*rotation
    coupling_derivative = derivative[:,None,None]*rotation+a[:,None,None]*theta_derivative[:,None,None]*angular_derivative
    covariance = coupling@coupling.transpose(0,2,1)
    covariance_derivative = coupling_derivative@coupling.transpose(0,2,1)+coupling@coupling_derivative.transpose(0,2,1)
    return x,coupling,coupling_derivative,covariance,covariance_derivative


def response(experiment, strength, order=512):
    contrast = experiment['contrast']
    x,coupling,derivative,covariance,covariance_derivative = geometry(contrast,experiment['winding'],experiment['twist'],order)
    a = np.sqrt(covariance[:,0,0])
    ratio = experiment['ratio']
    crossover = ratio/(1+ratio)
    if contrast == 0:
        potential_integral = -experiment['potential']*np.cos(x)/strength
    else:
        potential_integral = -experiment['potential']*np.cos(x)/(strength*a)
    log_density = (crossover-2)*np.log(a)+potential_integral
    density = np.exp(log_density-np.max(log_density))
    density /= np.sum(density)
    force = np.array([experiment['potential']*np.sin(x),np.full_like(x,experiment['push'])]).T
    drift = force+crossover*strength*np.einsum("nk,njk->nj",coupling[:,0,:],derivative)
    if experiment['observable'] == 'cosine':
        return float(density@np.cos(x))
    return float(density@drift[:,1])


def predict_at(experiments, strength):
    cache = {}
    values = []
    for experiment in experiments:
        key = tuple(sorted(experiment.items()))
        if key not in cache:
            cache[key] = response(experiment,strength)
        values.append(cache[key])
    return np.array(values)


class Model:
    def __init__(self):
        self.noise_strength = None

    def fit(self, records):
        inputs = [r['input'] for r in records]
        values = np.array([r['value'] for r in records])
        errors = np.array([r['sigma'] for r in records])
        def loss(strength):
            residual = (predict_at(inputs,strength)-values)/errors
            return float(residual@residual)
        result = minimize_scalar(loss,bounds=(.55,1.1),method='bounded',options={'xatol':1e-12})
        self.noise_strength = float(result.x)
        return self

    def predict(self, experiments):
        return predict_at(experiments,self.noise_strength)
