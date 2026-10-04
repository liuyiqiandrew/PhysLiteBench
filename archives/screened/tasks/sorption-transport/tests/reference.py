"""Conserved dissolved-plus-bound amounts and local equilibrium inversion."""
from functools import lru_cache
import numpy as np
from scipy.integrate import solve_ivp

TRUE_RATE = .63


def stored(c):
    weights = c*np.array([2., 1.])
    return c+4*weights/(1+weights.sum(axis=-1, keepdims=True))


def dissolved(total):
    free = np.full(total.shape[:-1], .2)
    low, high = np.zeros_like(free), np.ones_like(free)
    for iteration in range(40):
        denominator = 1+free[..., None]*np.array([8., 4.])
        residual = free+(total*np.array([2., 1.])*free[..., None]/denominator).sum(axis=-1)-1
        low = np.where(residual < 0, free, low)
        high = np.where(residual >= 0, free, high)
        derivative = 1+(total*np.array([2., 1.])/denominator**2).sum(axis=-1)
        step = free-residual/derivative
        if np.max(abs(residual)) < 2e-13:
            break
        free = np.where((step > low)&(step < high), step, (low+high)/2)
    return total/(1+free[..., None]*np.array([8., 4.]))


@lru_cache(maxsize=128)
def trajectory(initial, tolerance=1e-12):
    initial = np.asarray(initial).reshape(12, 2)
    def change(time, state):
        c = dissolved(state.reshape(12, 2))
        currents = (c[:-1]-c[1:])*np.array([1., 3.])
        rhs = np.zeros_like(c)
        rhs[:-1] -= currents
        rhs[1:] += currents
        return rhs.ravel()
    return solve_ivp(change, (0., 144.), stored(initial).ravel(), method='DOP853', max_step=.5,
                     rtol=tolerance, atol=tolerance*.01, dense_output=True).sol


def predict(experiments, rate=TRUE_RATE):
    result=[]
    for e in experiments:
        sol=trajectory(tuple(np.asarray(e['initial'], dtype=float).ravel()))
        c=dissolved(sol(rate*e['time']).reshape(12, 2))
        result.append(c[e['chamber'], e['species']])
    return np.array(result)


def readings(initial, times, species):
    return [dict(initial=np.asarray(initial).tolist(), time=float(t), chamber=j, species=s)
            for t in times for j in [0, 2, 5, 6, 9, 11] for s in species]


def calibration_inputs():
    result=[]
    for species in range(2):
        for level in [.45, 1.25]:
            c=np.zeros((12, 2))
            c[:6, species]=level
            result.extend(readings(c, [2., 8., 25., 65.], [species]))
    return result


def hidden_inputs():
    a=np.zeros((12, 2)); a[:, 0]=.24; a[:6, 1]=1.6
    b=np.zeros((12, 2)); b[:6, 0]=1.2; b[6:, 1]=1.4
    c=np.zeros((12, 2)); c[:4]=[1.3, .9]; c[4:8]=[.05, .2]; c[8:]=[.6, 1.5]
    return dict(competitor_release=readings(a, [3., 12., 40., 110.], [0, 1]),
                opposing_fronts=readings(b, [3., 12., 40., 110.], [0, 1]),
                mixed_storage=readings(c, [3., 12., 40., 110.], [0, 1]))
