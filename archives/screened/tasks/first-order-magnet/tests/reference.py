"""Global free-energy minimization without solving self-consistency roots."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import xlogy

PARAMETER = 'coupling'
TRUE_PARAMETER = .55
BOUNDS = (.3, .8)
FOUR_SPIN = 1.2


def free_energy(m, temperature, field, coupling):
    fractions = np.array([(1+m)/2, (1-m)/2])
    entropy_term = np.sum(xlogy(fractions, fractions), axis=0)
    return -coupling*m**2/2-FOUR_SPIN*m**4/4-field*m+temperature*entropy_term


def predict(experiments, coupling):
    result = []
    grid = np.linspace(-1., 1., 8001)
    for e in experiments:
        t, h = e['temperature'], e['field']
        energy = free_energy(grid, t, h, coupling)
        indices = np.flatnonzero((energy[1:-1] <= energy[:-2]) &
                                 (energy[1:-1] <= energy[2:]))+1
        candidates = [-1., 1.]
        for i in indices:
            fit = minimize_scalar(lambda m: free_energy(m, t, h, coupling),
                                  bounds=(grid[i-1], grid[i+1]), method='bounded',
                                  options={'xatol': 1e-13})
            candidates.append(float(fit.x))
        result.append(min(candidates, key=lambda m: free_energy(m, t, h, coupling)))
    return np.array(result)


def calibration_inputs():
    return [dict(temperature=t, field=float(h)) for t in [1.6, 2., 2.5]
            for h in np.r_[np.linspace(-.9, -.12, 16), np.linspace(.12, .9, 16)]]


def hidden_inputs():
    return {f'low_temperature_{t}': [dict(temperature=t, field=float(h))
            for h in np.r_[np.linspace(-.015, -.003, 12), np.linspace(.003, .015, 12)]]
            for t in [.7, .78, .84]}
