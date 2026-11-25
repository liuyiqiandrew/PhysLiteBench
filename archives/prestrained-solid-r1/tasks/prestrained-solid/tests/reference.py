import itertools
import numpy as np
from scipy.optimize import brentq

TRUE_PARAMETER = 1.17


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float); axis /= np.linalg.norm(axis)
    x, y, z = axis
    k = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def experiment(stretches, direction, branch, turn=0., axis=(1., 2., -.7)):
    r = rotation(axis, turn)
    return dict(deformation=(r @ np.diag(stretches)).tolist(),
                direction=(r @ np.asarray(direction, dtype=float)).tolist(), branch=branch)


def energy(f):
    f = np.asarray(f, dtype=np.longdouble)
    c = f.T @ f
    i1 = np.trace(c)
    i2 = (i1 * i1 - np.trace(c @ c)) / 2
    volume = np.dot(f[0], np.cross(f[1], f[2]))
    logarithm = np.log(volume)
    return .3 * (i1 - 3) + .2 * (i2 - 3) - 1.4 * logarithm + logarithm * logarithm


def energy_operator(e, step=.001):
    f = np.asarray(e['deformation'], dtype=np.longdouble)
    n = np.asarray(e['direction'], dtype=np.longdouble); n /= np.sqrt(n @ n)
    v = f.T @ n
    zero = energy(f)
    def quadratic(a, h):
        change = np.outer(np.asarray(a, dtype=np.longdouble), v)
        return (-energy(f + 2*h*change) + 16*energy(f + h*change) - 30*zero
                + 16*energy(f - h*change) - energy(f - 2*h*change)) / (12*h*h)
    def refined(a):
        return float((16*quadratic(a, step/2) - quadratic(a, step)) / 15)
    basis = np.eye(3)
    result = np.empty((3, 3))
    for i in range(3):
        result[i, i] = refined(basis[i])
    for i in range(3):
        for j in range(i):
            result[i, j] = result[j, i] = (refined(basis[i] + basis[j]) - result[i, i] - result[j, j])/2
    return result


def predict(experiments, modulus=TRUE_PARAMETER, step=.001):
    return np.asarray([np.sqrt(1000. * modulus * np.linalg.eigvalsh(energy_operator(e, step))[e['branch']]) for e in experiments])


def lateral_stretch(axial):
    def equation(a):
        volume = a*a*axial
        return .6*a + .4*(a*a + axial*axial)*a + (2*np.log(volume) - 1.4)/a
    return brentq(equation, .8, 1.1, xtol=1e-14)


def calibration_inputs():
    result = []
    for axial, azimuth, branch, repeat in itertools.product([1., 1.08, 1.16, 1.24, 1.32, 1.4], [0., .6, 1.3], range(3), range(4)):
        a = lateral_stretch(axial)
        result.append(experiment([a, a, axial], [np.cos(azimuth), np.sin(azimuth), 0.], branch,
                                 turn=.23*repeat - .31))
    return result


def hidden_inputs():
    axial = []
    for b, branch in itertools.product([1.2, 1.3, 1.4], range(3)):
        a = lateral_stretch(b)
        axial.append(experiment([a, a, b], [.2, -.1, 1.], branch, .47))
    mixed = []
    for stretches, direction, turn in [([1.35, .92, .87], [1., .1, -.2], -.38),
                                       ([1.25, 1.03, .88], [.7, -.2, .4], .63),
                                       ([1.18, 1.05, .9], [.8, .6, -.1], -.71)]:
        for branch in range(3):mixed.append(experiment(stretches, direction, branch, turn))
    volume = []
    for stretches, direction, turn in [([1.04, 1.04, 1.04], [.3, -.7, .6], .2),
                                       ([1.055, 1.055, 1.055], [-.8, .2, .5], -.6),
                                       ([1.15, 1.05, .96], [.9, -.1, .2], .7)]:
        for branch in range(3):volume.append(experiment(stretches, direction, branch, turn))
    return {'axial_preload': axial, 'mixed_stretches': mixed, 'volume_change': volume}
