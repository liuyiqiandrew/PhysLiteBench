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
    contrast = experiment['contrast']
    wave = experiment['wavenumber']
    temperature = base*(1+contrast*np.cos(wave*x))
    gradient = -base*contrast*wave*np.sin(wave*x)
    force = experiment['force_x']
    if contrast == 0:
        density = np.full(points, 1/(2*np.pi))
        current = force/(2*np.pi*friction)
    else:
        # Constant flux: friction*J = force*rho - d(T*rho)/dx.
        system = np.zeros((points+1,points+1))
        system[:points,:points] = force*np.eye(points)-derivative*temperature[None,:]
        system[:points,points] = -1
        system[points,:points] = 2*np.pi/points
        rhs = np.zeros(points+1); rhs[-1] = 1
        state = np.linalg.solve(system,rhs)
        density = state[:points]
        current = state[-1]/friction
    return temperature, gradient, density, current, 2*np.pi/points


@lru_cache(None)
def entropy_coefficient(force_x, force_y, drag_ratio, temperature, contrast, wavenumber):
    e = dict(force_x=force_x,force_y=force_y,drag_ratio=drag_ratio,
             temperature=temperature,contrast=contrast,wavenumber=wavenumber)
    # The normalized positional density is independent of the overall drag
    # scale. Compute the coefficient of 1/friction at unit x friction.
    temp, gradient, density, current_x, dx = positional_state(e,1.)
    current_y = force_y*density/drag_ratio
    positional = np.sum(current_x**2/(temp*density)
                        +drag_ratio*current_y**2/(temp*density))*dx
    heat_coefficient = .5*(1+1/(1+2*drag_ratio))
    return float(positional+heat_coefficient*np.sum(density*gradient**2/temp)*dx)


def coefficients(experiments):
    return np.array([entropy_coefficient(e['force_x'],e['force_y'],e['drag_ratio'],
                     e['temperature'],e['contrast'],e['wavenumber']) for e in experiments])


class Model:
    def __init__(self):
        self.friction = None

    def fit(self, records):
        design = coefficients([r['input'] for r in records])
        values = np.array([r['value'] for r in records])
        variance = np.array([r['sigma']**2 for r in records])
        mobility = np.sum(design*values/variance)/np.sum(design**2/variance)
        self.friction = float(np.clip(1/mobility,.7,1.6))
        return self

    def predict(self, experiments):
        return coefficients(experiments)/self.friction
