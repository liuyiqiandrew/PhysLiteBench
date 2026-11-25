from functools import lru_cache
import numpy as np
from scipy.optimize import brentq


def pressure(volume, temperature, attraction):
    return temperature/(volume-1)-attraction/volume**2


@lru_cache(None)
def coexistence(temperature, attraction):
    spinodals = np.sort(np.roots([temperature, -2*attraction, 4*attraction, -2*attraction]).real)
    low, high = spinodals[-2:]
    p_min = pressure(low, temperature, attraction)
    p_max = pressure(high, temperature, attraction)
    def volumes(p):
        liquid = brentq(lambda v: pressure(v, temperature, attraction)-p, 1+1e-10, low)
        vapor = brentq(lambda v: pressure(v, temperature, attraction)-p,
                      high, max(2*high, 2*temperature/p))
        return liquid, vapor
    def area(p):
        liquid, vapor = volumes(p)
        return (temperature*np.log((vapor-1)/(liquid-1))
                +attraction*(1/vapor-1/liquid)-p*(vapor-liquid))
    lower = max(0., p_min)+1e-9*(p_max-max(0., p_min))
    upper = p_max-1e-9*(p_max-max(0., p_min))
    p = brentq(area, lower, upper, xtol=1e-13)
    return (*volumes(p), p)


class Model:
    def __init__(self):
        self.attraction = None

    def fit(self, records):
        v = np.array([r['input']['volume'] for r in records])
        t = np.array([r['input']['temperature'] for r in records])
        value = np.array([r['value'] for r in records])
        weight = 1/np.array([r['sigma'] for r in records])**2
        x = 1/v**2
        self.attraction = float(np.sum(weight*x*(t/(v-1)-value))/np.sum(weight*x*x))
        return self

    def predict(self, experiments):
        out = []
        critical_temperature = 8*self.attraction/27
        for e in experiments:
            t, v = e['temperature'], e['volume']
            p = pressure(v, t, self.attraction)
            # Inside this tiny critical window the pressure loop is below
            # 1e-10; resolving three nearly coincident roots loses precision.
            if t < critical_temperature*(1-1e-7):
                liquid, vapor, equilibrium_pressure = coexistence(t, self.attraction)
                if liquid <= v <= vapor:
                    p = equilibrium_pressure
            out.append(p)
        return np.array(out)
