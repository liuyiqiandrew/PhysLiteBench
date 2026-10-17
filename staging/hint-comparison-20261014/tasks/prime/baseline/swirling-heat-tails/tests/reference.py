"""Gaussian path precision and stationary/terminal integration."""
import numpy as np
from scipy.optimize import minimize_scalar

TRUE_PARAMETER = 1.08


def experiment(stiffness=1., drive_ratio=.3, relative_heat=6.):
    return dict(stiffness=stiffness, drive_ratio=drive_ratio, relative_heat=relative_heat)


def calibration_inputs():
    return [experiment(stiffness=k, drive_ratio=r, relative_heat=z)
            for k in [.7, 1.4] for r in [.1, .3, .7, 1.2] for z in [.4, .7, 1.3, 1.6]]


def hidden_inputs():
    return {
        'central_anchors': [experiment(stiffness=k, drive_ratio=r, relative_heat=z)
                            for k in [.5, 2.] for r in [.15, .55, 1.1] for z in [-.5, .2, 1.8, 2.]],
        'reverse_weak_drive': [experiment(stiffness=k, drive_ratio=r, relative_heat=z)
                               for k in [.6, 1.8] for r in [.1, .25, .45] for z in [-2., -4., -6.]],
        'reverse_strong_drive': [experiment(stiffness=k, drive_ratio=r, relative_heat=z)
                                 for k in [.5, 2.] for r in [.7, .9, 1.2] for z in [-3., -6.]],
        'forward_tails': [experiment(stiffness=k, drive_ratio=r, relative_heat=z)
                          for k in [.6, 1.7] for r in [.1, .2, .3] for z in [6., 9.]],
    }


def predict(experiments, mobility, step=.006):
    result = []
    for e in experiments:
        rates = [discrete_rate(e['drive_ratio'], e['relative_heat'], h) for h in [step, step/2, step/4]]
        extrapolated = (rates[0]-6*rates[1]+8*rates[2])/3
        result.append(mobility*e['stiffness']*extrapolated)
    return np.asarray(result)


def coefficients(r, lam, step):
    decay = np.exp(-step)
    variance = -np.expm1(-2*step)
    cross = (decay*np.cos(r*step)/variance)**2+(decay*np.sin(r*step)/variance+lam*r)**2
    diagonal = (1+decay**2)/variance
    disc = 1-8*lam*r*decay*np.sin(r*step)/variance-4*lam**2*r**2
    if disc < 0:
        return None
    plus, minus = (diagonal+np.sqrt(disc))/2, (diagonal-np.sqrt(disc))/2
    initial = 1+decay**2/variance-lam
    final = 1/variance+lam-cross/plus
    if initial <= minus or final <= 0:
        return None
    return variance, plus, cross, diagonal, initial, 1/variance+lam


def discrete_scgf(r, lam, step):
    values = coefficients(r, lam, step)
    if values is None:
        return np.inf
    variance, plus, *_ = values
    return -np.log(variance*plus)/step


def domain(r, step):
    result = []
    for direction in [-1, 1]:
        good, bad = 0., 8.
        for _ in range(55):
            middle = (good+bad)/2
            if np.isfinite(discrete_scgf(r, direction*middle, step)):
                good = middle
            else:
                bad = middle
        result.append(direction*good)
    return result


def discrete_rate(r, z, step):
    low, high = domain(r, step)
    q = 2*r*r*z
    opt = minimize_scalar(lambda lam: discrete_scgf(r, lam, step)-lam*q,
                          bounds=(low+1e-10, high-1e-10), method='bounded', options={'xatol':1e-12})
    candidates = [opt.fun]+[discrete_scgf(r, lam, step)-lam*q for lam in [low+1e-10, high-1e-10]]
    return -min(candidates)


def finite_time_scgf(r, lam, step, time):
    values = coefficients(r, lam, step)
    if values is None:
        # Still build the finite-chain Gaussian integral outside the asymptotic domain.
        decay = np.exp(-step)
        variance = -np.expm1(-2*step)
        cross = (decay*np.cos(r*step)/variance)**2+(decay*np.sin(r*step)/variance+lam*r)**2
        diagonal = (1+decay**2)/variance
        initial = 1+decay**2/variance-lam
        final = 1/variance+lam
    else:
        variance, _, cross, diagonal, initial, final = values
    count = int(round(time/step))
    pivot = initial
    logdet = np.log(pivot)
    for i in range(1, count+1):
        pivot = (final if i == count else diagonal)-cross/pivot
        if pivot <= 0:
            return np.inf
        logdet += np.log(pivot)
    return (-count*np.log(variance)-logdet)/(count*step)


