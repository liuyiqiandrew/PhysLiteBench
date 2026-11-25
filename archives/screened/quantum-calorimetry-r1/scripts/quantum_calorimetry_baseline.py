import numpy as np
from scipy.integrate import quad_vec
from functools import lru_cache


def mode_capacity(frequency, temperature):
    x = np.asarray(frequency)/temperature
    decay = np.exp(-x)
    return x*x*decay/(-np.expm1(-x))**2


def susceptibility(omega, frequency, damping, cutoff):
    return 1/(frequency**2-omega**2
              -1j*omega*damping*cutoff/(cutoff-1j*omega))


@lru_cache(None)
def stationary_moments(temperature, damping, cutoff, frequency):
    if damping == 0:
        factor = 1/np.tanh(frequency/(2*temperature))
        capacity = float(mode_capacity(frequency, temperature))
        return np.array([factor/(2*frequency), frequency*factor/2,
                         capacity/frequency**2, capacity])
    def integrand(omega):
        imaginary = susceptibility(omega, frequency, damping, cutoff).imag
        factor = 1/np.tanh(omega/(2*temperature))
        derivative = 2*float(mode_capacity(omega, temperature))/omega
        return imaginary/np.pi*np.array([factor, omega**2*factor,
                                         derivative, omega**2*derivative])
    return quad_vec(integrand, 0., np.inf, epsabs=2e-10,
                    epsrel=2e-10, points=[frequency])[0]


def heat_capacity(experiment, frequency):
    q2, p2, dq2, dp2 = stationary_moments(
        experiment['temperature'], experiment['damping'],
        experiment['cutoff'], frequency)
    return float((dp2+frequency**2*dq2)/2)


def predict_at(experiments, frequency):
    return np.array([heat_capacity(e, frequency) for e in experiments])


class Model:
    def __init__(self):
        self.natural_frequency = None

    def fit(self, records):
        from scipy.optimize import minimize_scalar
        temperatures = np.array([r['input']['temperature'] for r in records])
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def loss(frequency):
            residual = (mode_capacity(frequency, temperatures)-values)/sigma
            return float(residual@residual)
        self.natural_frequency = float(minimize_scalar(
            loss, bounds=(.8,1.2), method='bounded',
            options={'xatol':1e-12}).x)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.natural_frequency)
