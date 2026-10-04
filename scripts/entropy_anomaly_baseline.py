from functools import lru_cache
import numpy as np


@lru_cache(None)
def spatial_grid(points):
    x = np.arange(points)*2*np.pi/points
    modes = np.fft.fftfreq(points, 1/points)
    derivative = np.fft.ifft(1j*modes[:,None]*np.fft.fft(np.eye(points),axis=0),axis=0).real
    return x, derivative


def positional_state(experiment, friction, points=129):
    x, derivative = spatial_grid(points)
    base = experiment['temperature']
    temperature = base*(1+experiment['contrast']*np.cos(experiment['wavenumber']*x))
    gradient = -base*experiment['contrast']*experiment['wavenumber']*np.sin(experiment['wavenumber']*x)
    gamma_y = friction*experiment['drag_ratio']
    magnetic = experiment['magnetic']
    effective_force = experiment['force_x']+magnetic*experiment['force_y']/gamma_y
    effective_drag = friction+magnetic**2/gamma_y
    system = np.zeros((points+1,points+1))
    system[:points,:points] = effective_force*np.eye(points)-derivative*temperature[None,:]
    system[:points,points] = -effective_drag
    system[points,:points] = 2*np.pi/points
    rhs = np.zeros(points+1); rhs[-1] = 1
    state = np.linalg.solve(system,rhs)
    density = state[:points]
    current_x = state[-1]
    current_y = (experiment['force_y']*density-magnetic*current_x)/gamma_y
    return temperature, gradient, density, current_x, current_y, 2*np.pi/points


@lru_cache(None)
def entropy_rate(force_x, force_y, drag_ratio, magnetic, temperature, contrast, wavenumber, friction):
    e = dict(force_x=force_x,force_y=force_y,drag_ratio=drag_ratio,magnetic=magnetic,
             temperature=temperature,contrast=contrast,wavenumber=wavenumber)
    temp, gradient, density, current_x, current_y, dx = positional_state(e,friction)
    gamma_y = friction*drag_ratio
    positional = np.sum((friction*current_x**2+gamma_y*current_y**2)/(temp*density))*dx
    heat_coefficient = .5*(1/friction+1/(friction+2*gamma_y))
    return float(positional+heat_coefficient*np.sum(density*gradient**2/temp)*dx)


def coefficients(experiments, friction=1.):
    return np.array([entropy_rate(e['force_x'],e['force_y'],e['drag_ratio'],e['magnetic'],
                     e['temperature'],e['contrast'],e['wavenumber'],friction) for e in experiments])


class Model:
    def __init__(self):
        self.friction = None

    def fit(self, records):
        inputs = [r['input'] for r in records]
        value = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        coefficient = coefficients(inputs)
        inverse_friction = np.sum(coefficient*value/sigma**2)/np.sum(coefficient**2/sigma**2)
        self.friction = float(1/inverse_friction)
        return self

    def predict(self, experiments):
        return coefficients(experiments,self.friction)
