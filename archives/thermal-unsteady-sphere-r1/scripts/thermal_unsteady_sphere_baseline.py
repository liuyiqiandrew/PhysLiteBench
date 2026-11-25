"""Sphere velocity response and fluctuations."""
import numpy as np
from scipy.optimize import minimize_scalar

KB = 1.380649e-23
FLUID_DENSITY = 1000.0
PARTICLE_DENSITY = 2200.0
FORCE = 1e-13


def mobility(radius, angular_frequency, viscosity):
    a = radius*1e-6
    omega = angular_frequency*1e6
    eta = viscosity*1e-3
    mass = 4*np.pi*PARTICLE_DENSITY*a**3/3
    s = a*np.sqrt(-1j*omega*FLUID_DENSITY/eta)
    impedance = 6*np.pi*eta*a*(1+s+s*s/9)
    return 1/(impedance-1j*omega*mass)


def predict_at(experiments, viscosity):
    readings = []
    for experiment in experiments:
        a = experiment['radius']*1e-6
        eta = viscosity*1e-3
        mu = mobility(experiment['radius'], experiment['angular_frequency'], viscosity)
        if experiment['observable'] == 'in_phase':
            readings.append(1e6*FORCE*mu.real)
        elif experiment['observable'] == 'quadrature':
            readings.append(1e6*FORCE*mu.imag)
        else:
            resistance = 6*np.pi*eta*a
            force_spectrum = 2*KB*experiment['temperature']*resistance
            readings.append(1e12*force_spectrum*abs(mu)**2)
    return np.array(readings)


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        inputs = [record['input'] for record in records]
        values = np.array([record['value'] for record in records])
        sigma = np.array([record['sigma'] for record in records])
        def objective(viscosity):
            residual = (predict_at(inputs, viscosity)-values)/sigma
            return float(residual@residual)
        result = minimize_scalar(objective, bounds=(.7,1.4), method='bounded',
                                 options={'xatol':1e-12})
        self.viscosity = float(min([result.x,.7,1.4], key=objective))
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise RuntimeError('Call fit before predict.')
        return predict_at(experiments, self.viscosity)
