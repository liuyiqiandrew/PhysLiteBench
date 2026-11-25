import numpy as np

from scipy.optimize import minimize_scalar
from scipy.special import xlogy

def predict(experiments, coupling):
    result = []
    for e in experiments:
        def free_energy(m):
            up, down = (1+m)/2, (1-m)/2
            return -.5*coupling*m*m-e['field']*m+e['temperature']*(xlogy(up, up)+xlogy(down, down))
        grid = np.linspace(-1, 1, 4001)
        i = int(np.argmin(free_energy(grid)))
        fit = minimize_scalar(free_energy, bounds=(grid[max(0, i-1)], grid[min(4000, i+1)]),
                              method='bounded', options={'xatol': 1e-13})
        result.append(fit.x)
    return np.array(result)


def calibration_inputs():
    return [dict(temperature=t, field=float(h)) for t in [1.5, 2., 3.]
        for h in np.r_[np.linspace(-.8, -.08, 16), np.linspace(.08, .8, 16)]]

def hidden_inputs():
    return {f'low_temperature_{t}': [dict(temperature=t, field=float(h))
           for h in np.r_[np.linspace(-.04, -.002, 12), np.linspace(.002, .04, 12)]] for t in [.48, .6, .7]}

PARAMETER = 'coupling'
TRUE_PARAMETER = 0.86
BOUNDS = (0.3, 1.2)
