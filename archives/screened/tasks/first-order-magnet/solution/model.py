import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import xlogy

QUARTIC_COUPLING = 1.2


def free_energy(m, temperature, field, coupling):
    up, down = (1+m)/2, (1-m)/2
    return (-coupling*m*m/2-QUARTIC_COUPLING*m**4/4-field*m
            +temperature*(xlogy(up, up)+xlogy(down, down)))


def equilibrium(temperature, field, coupling):
    def equation(m):
        return m-np.tanh((coupling*m+QUARTIC_COUPLING*m**3+field)/temperature)
    # Above this sufficient convexity bound, the stationary point is unique.
    convex_bound = max(coupling, (coupling+3*QUARTIC_COUPLING)**2/(12*QUARTIC_COUPLING))
    if temperature > convex_bound:
        return brentq(equation, -1., 1., xtol=1e-13)
    # Zeros of f'' split f' into monotone intervals; bracket every stationary point.
    squared_turns = np.roots([3*QUARTIC_COUPLING,
                             coupling-3*QUARTIC_COUPLING, temperature-coupling])
    turns = [np.sqrt(float(y.real)) for y in squared_turns
             if abs(y.imag) < 1e-12 and 0 < y.real < 1]
    grid = np.array(sorted([-1., 1., *turns, *[-x for x in turns]]))
    values = equation(grid)
    candidates = list(grid[values == 0])
    for i in np.flatnonzero(values[:-1]*values[1:] < 0):
        candidates.append(brentq(equation, grid[i], grid[i+1], xtol=1e-13))
    return min(candidates, key=lambda m: free_energy(m, temperature, field, coupling))


class Model:
    def __init__(self):
        self.coupling = None

    def fit(self, records):
        experiments = [r['input'] for r in records]
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def loss(coupling):
            self.coupling = coupling
            return np.sum(((self.predict(experiments)-values)/sigma)**2)
        answer = minimize_scalar(loss, bounds=(.3, .8), method='bounded',
                                 options={'xatol': 1e-11})
        self.coupling = float(answer.x)
        return self

    def predict(self, experiments):
        return np.array([equilibrium(e['temperature'], e['field'], self.coupling)
                         for e in experiments])
