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
    force = experiment['force']
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


class Model:
    def __init__(self):
        self.friction = None

    def fit(self, records):
        coefficient = np.array([r['input']['force']**2/r['input']['temperature'] for r in records])
        values = np.array([r['value'] for r in records])
        variance = np.array([r['sigma']**2 for r in records])
        mobility = np.sum(coefficient*values/variance)/np.sum(coefficient**2/variance)
        self.friction = float(np.clip(1/mobility,.7,1.6))
        return self

    def predict(self, experiments):
        out = []
        for e in experiments:
            temperature, gradient, density, current, dx = positional_state(e,self.friction)
            diffusion = temperature/self.friction
            value = np.sum(current**2/(diffusion*density))*dx
            out.append(float(value))
        return np.array(out)
