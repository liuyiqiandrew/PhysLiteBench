from functools import lru_cache
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

SITES = 4.0
AFFINITY = np.array([2., 1.])
RELATIVE_RATE = np.array([1., 3.])


def laplacian(c):
    flow = np.diff(c, axis=0)*RELATIVE_RATE
    change = np.zeros_like(c)
    change[:-1] += flow
    change[1:] -= flow
    return change


@lru_cache(maxsize=128)
def trajectory(initial):
    initial = np.array(initial).reshape(12, 2)
    def rhs(time, state):
        c = state.reshape(12, 2)
        flux = laplacian(c)
        z = 1+c@AFFINITY
        aa = 1+SITES*AFFINITY[0]*(1+AFFINITY[1]*c[:, 1])/z**2
        bb = 1+SITES*AFFINITY[1]*(1+AFFINITY[0]*c[:, 0])/z**2
        ab = -SITES*np.prod(AFFINITY)*c[:, 0]/z**2
        ba = -SITES*np.prod(AFFINITY)*c[:, 1]/z**2
        determinant = aa*bb-ab*ba
        derivative = np.column_stack(((bb*flux[:, 0]-ab*flux[:, 1])/determinant,
                                      (aa*flux[:, 1]-ba*flux[:, 0])/determinant))
        return derivative.ravel()
    return solve_ivp(rhs, (0., 144.), initial.ravel(), method='DOP853', max_step=.5,
                     rtol=2e-11, atol=2e-13, dense_output=True).sol


class Model:
    def __init__(self):
        self.rate = None

    def fit(self, records):
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        experiments = [r['input'] for r in records]
        def residual(parameter):
            self.rate = float(parameter[0])
            return (self.predict(experiments)-values)/sigma
        result = least_squares(residual, [.5], bounds=(.1, 1.2),
                               ftol=1e-11, xtol=1e-11, gtol=1e-11)
        self.rate = float(result.x[0])
        return self

    def predict(self, experiments):
        result = []
        for e in experiments:
            sol = trajectory(tuple(np.asarray(e['initial'], dtype=float).ravel()))
            result.append(sol(self.rate*e['time']).reshape(12, 2)[e['chamber'], e['species']])
        return np.array(result)
