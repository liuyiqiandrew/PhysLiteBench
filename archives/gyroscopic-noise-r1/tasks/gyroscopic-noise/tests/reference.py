"""Independent stationary Langevin time-correlation reference."""
import numpy as np
from scipy.linalg import solve_continuous_lyapunov

TRUE_PARAMETER = .67


def dynamics(experiment, drag):
    theta = experiment['trap_angle']
    c, s = np.cos(theta), np.sin(theta)
    k1, k2 = experiment['stiffness_1'], experiment['stiffness_2']
    k = np.array([[k1*c*c+k2*s*s, (k1-k2)*c*s],
                  [(k1-k2)*c*s, k1*s*s+k2*c*c]])
    field = experiment['field']
    a = np.zeros((4, 4))
    a[:2, 2:] = np.eye(2)
    a[2:, :2] = -k
    a[2:, 2:] = [[-drag, field], [-field, -drag]]
    q = np.zeros((4, 4))
    q[2:, 2:] = 2*drag*experiment['temperature']*np.eye(2)
    return a, q, k


def position_spectrum(experiment, drag):
    a, q, _ = dynamics(experiment, drag)
    covariance = solve_continuous_lyapunov(a, -q)
    # The positive-time covariance is exp(A*t) C. Integrate it and
    # its negative-time transpose using two independent resolvents.
    z = -a-1j*experiment['frequency']*np.eye(4)
    positive = np.linalg.solve(z, covariance)
    total = positive+positive.conj().T
    return total[:2, :2]


def predict(experiments, drag):
    values = []
    for e in experiments:
        spectral = position_spectrum(e, drag)
        phase = e['frequency']*e['delay']
        # Expanded detector contraction, independently of the model helper.
        value = (spectral[0, 0].real+e['weight']**2*spectral[1, 1].real
                 +2*e['weight']*(np.cos(phase)*spectral[0, 1].real
                                  +np.sin(phase)*spectral[0, 1].imag))
        values.append(value)
    return np.array(values)


def experiment(frequency, field, angle, weight, delay, temperature=1.,
               k1=1., k2=2.1):
    return dict(frequency=frequency, field=field, trap_angle=angle,
                weight=weight, delay=delay, temperature=temperature,
                stiffness_1=k1, stiffness_2=k2)


def calibration_inputs():
    return [experiment(*args) for args in [
        (0., 1.2, .3, .8, 0., .8),
        (0., -.7, -.6, -.7, 0., 1.2),
        (.25, 1.1, .2, .6, 0., 1.),
        (.45, -.9, -.5, -.8, 0., .7),
        (.65, 1.3, .7, 1.1, 0., 1.1),
        (.85, .6, -.3, -.6, 0., .9),
        (1.05, -.8, .4, .9, 0., 1.3),
        (1.2, 1.1, -.8, -.9, 0., .8),
        (1.4, -.7, .8, .5, 0., 1.2),
        (1.65, 1.4, .1, -.5, 0., 1.),
        (1.9, -.6, -.4, .8, 0., .6),
        (2.15, 1.2, .6, -1.1, 0., 1.1),
    ]]


def hidden_inputs():
    return {
        'positive_bias': [experiment(*args) for args in [
            (.7, 1.2, .2, .8, 1.8), (.5, 1.3, .4, 1., 2.5),
            (1.1, .8, .6, 1., 1.4), (.85, 1.5, -.5, .9, 2.1)]],
        'negative_bias': [experiment(*args) for args in [
            (.9, -1.3, -.3, .7, 2.), (.6, -1.1, .6, .9, 2.2),
            (1.2, -1.5, -.7, 1.2, 1.4), (.8, -.9, .3, -.8, 2.4)]],
        'rotated_traps': [experiment(*args) for args in [
            (.75, 1.1, 1.2, .9, 2.1, .6, .8, 2.4),
            (1.1, -1.4, -.9, -.8, 1.7, 1.3, 1.2, 1.7),
            (.6, .9, -.8, 1.1, 2.8, .9, .7, 2.2),
            (1.25, 1.2, .9, -1., 1.2, 1.1, 1.3, 2.5)]],
        'instantaneous_and_reciprocal': [
            experiment(.6, 1.3, .7, .8, 0.),
            experiment(1.2, -1.2, -.3, -.7, 0.),
            experiment(.8, 0., .5, .9, 2.),
            experiment(1.5, 0., -.7, -.8, 1.2),
            experiment(0., 1.1, .4, .6, 2.4),
        ],
    }
