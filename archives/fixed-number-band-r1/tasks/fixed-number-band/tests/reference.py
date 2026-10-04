"""Finite canonical partition sums, followed by thermodynamic extrapolation."""
from functools import lru_cache
import numpy as np
from scipy.special import gammaln, logsumexp

TRUE_PARAMETER = 1.07


@lru_cache(maxsize=4096)
def canonical(width, temperature, filling, band_size):
    m = int(band_size)
    number = int(round(2*m*filling))
    upper = np.arange(max(0, number-m), min(m, number)+1, dtype=float)
    lower = number-upper
    log_weight = (2*gammaln(m+1)-gammaln(upper+1)-gammaln(m-upper+1)
                  -gammaln(lower+1)-gammaln(m-lower+1)-width*upper/temperature)
    probability = np.exp(log_weight-logsumexp(log_weight))
    mean = float(np.sum(probability*upper))
    variance = float(np.sum(probability*(upper-mean)**2))
    energy = width*(mean-number/2)/(2*m)
    capacity = width**2*variance/(2*m*temperature**2)
    return float(energy), float(capacity), number/(2*m)


def predict(experiments, width, base_size=8000):
    output = []
    for e in experiments:
        values = [canonical(float(width), float(e['temperature']), float(e['filling']), m)[1]
                  for m in (base_size, 2*base_size, 4*base_size)]
        output.append((values[0]-6*values[1]+8*values[2])/3)
    return np.array(output)


def calibration_inputs():
    return [{'temperature':float(t), 'filling':.5} for t in np.linspace(.1,.15,9)]


def hidden_inputs():
    return {
        'low_fillings': [{'temperature':t,'filling':n} for t,n in [( .4,.2),(.45,.25),(.5,.3),(.6,.2)]],
        'high_fillings':[{'temperature':t,'filling':n} for t,n in [(.4,.8),(.45,.75),(.5,.7),(.6,.8)]],
        'temperature_sweep':[{'temperature':t,'filling':n} for t,n in [(.4,.25),(.45,.3),(.55,.25),(.6,.3)]],
        'half_filling_anchor':[{'temperature':t,'filling':.5} for t in [.12,.2,.35,.55]],
    }
