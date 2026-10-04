from functools import lru_cache
import numpy as np


def _rates(experiment, friction):
    return _rates_at(experiment['temperature'], experiment['contrast'], experiment['wavenumber'], friction)


@lru_cache(256)
def _rates_at(base, contrast, wave, friction):
    x = np.arange(512)*2*np.pi/512
    temperature = base*(1+contrast*np.cos(x))
    gradient = -base*contrast*wave*np.sin(x)
    density = 1/temperature
    density /= np.sum(density)
    local_rate = gradient**2/(2*friction*temperature)
    mean = float(density@local_rate)
    rhs = -friction*(local_rate-mean)/(wave**2*temperature)
    modes = np.fft.fftfreq(len(x), 1/len(x))
    transformed = np.fft.fft(rhs)
    potential = np.zeros(len(x), dtype=complex)
    potential[1:] = -transformed[1:]/modes[1:]**2
    potential = np.fft.ifft(potential).real
    potential -= density@potential
    variance = float(2*density@((local_rate-mean)*potential))
    return mean, variance


class Model:
    def __init__(self):
        self.friction = 1.0

    def fit(self, records):
        inputs = [r['input'] for r in records]
        unit = np.asarray([_rates(e, 1.0)[0] for e in inputs])
        measured = np.asarray([r['value'] for r in records])
        sigma = np.asarray([r['sigma'] for r in records])
        inverse = np.sum(unit*measured/sigma**2)/np.sum(unit**2/sigma**2)
        self.friction = float(np.clip(1/inverse, .7, 1.6))
        return self

    def predict(self, experiments):
        values = []
        for e in experiments:
            mean, variance = _rates(e, self.friction)
            values.append(mean if e['statistic'] == 'mean' else variance)
        return np.asarray(values, dtype=float)
