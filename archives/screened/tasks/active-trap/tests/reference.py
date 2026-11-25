"""Stationary Cartesian polynomial moments, independent of angular evolution."""
from functools import lru_cache
import math
import numpy as np
from scipy.integrate import quad

TRUE_PARAMETER = .65


def radial_moments(trap, rotational_diffusion, order=36):
    # Moments of z = trap*(x+iy)/speed with translational noise removed.
    # Rotational symmetry fixes the orientation exponent to q-p.
    @lru_cache(None)
    def moment(p, q):
        if p+q == 0:
            return 1.
        numerator = 0.
        if p:
            numerator += trap*p*moment(p-1, q)
        if q:
            numerator += trap*q*moment(p, q-1)
        return numerator/(trap*(p+q)+rotational_diffusion*(q-p)**2)
    return np.array([moment(n, n) for n in range(order+1)])


def characteristic(experiment, rotational_diffusion, order=36):
    trap = float(experiment['trap_rate'])
    q = float(experiment['wavenumber'])
    speed = float(experiment['speed'])
    thermal = float(experiment['diffusion'])
    moments = radial_moments(trap, rotational_diffusion, order)
    argument = (q*speed/trap)**2/4
    terms = [(-argument)**n*moments[n]/math.factorial(n)**2 for n in range(order+1)]
    return math.fsum(terms)*np.exp(-thermal*q*q/(2*trap))


def predict_one(experiment, rotational_diffusion):
    speed = float(experiment['speed'])
    diffusion = float(experiment['diffusion'])
    if experiment['readout'] == 'free_msd':
        time = float(experiment['time'])
        correlation_integral = quad(lambda lag: (time-lag)*np.exp(-rotational_diffusion*lag),
                                    0, time, epsabs=1e-12, epsrel=1e-12)[0]
        return 4*diffusion*time+2*speed**2*correlation_integral
    if experiment['readout'] == 'trap_variance':
        trap = float(experiment['trap_rate'])
        moment = radial_moments(trap, rotational_diffusion, 1)[1]
        return diffusion/trap + .5*(speed/trap)**2*moment
    return characteristic(experiment, rotational_diffusion)


def predict(experiments, rotational_diffusion=TRUE_PARAMETER):
    return np.array([predict_one(e, rotational_diffusion) for e in experiments])


def calibration_inputs():
    experiments = []
    for speed, diffusion in [(1., .02), (1.7, .06), (2.2, .1)]:
        for time in np.geomspace(.15, 7.5, 20):
            experiments.append({'readout': 'free_msd', 'speed': speed,
                                'diffusion': diffusion, 'time': float(time)})
        for trap in np.linspace(.45, 2.4, 12):
            experiments.append({'readout': 'trap_variance', 'speed': speed,
                                'diffusion': diffusion, 'trap_rate': float(trap)})
    return experiments


def hidden_inputs():
    groups = {}
    for name, trap, speed, diffusion, arguments in [
        ('persistent_cloud', 2.2, 1.8, .012, np.linspace(2.5, 5.5, 9)),
        ('moderate_trap', .85, 1.3, .018, np.linspace(3., 6.5, 9)),
        ('thermal_blurring', 1.5, 2.1, .07, np.linspace(2.7, 5.8, 9))]:
        groups[name] = [{'readout': 'trap_fourier', 'speed': speed, 'diffusion': diffusion,
                         'trap_rate': trap, 'wavenumber': float(a*trap/speed)} for a in arguments]
    return groups
