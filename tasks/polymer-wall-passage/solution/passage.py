from scipy.special import hyp2f1
import numpy as np

H = 5.0 / 8


def unconditioned(z, w):
    rz, rw = (1 + z.real / abs(z)) / 2, (1 + w.real / abs(w)) / 2
    s = abs((z - w) / (z - w.conjugate())) ** 2
    shape = 1 - s * hyp2f1(1, 4/3, 5/3, 1-s)
    both = rz * rw + z.imag * w.imag * shape / (4 * abs(z) * abs(w))
    return np.array([1., rz, rw, both])


def density(z, w, walls):
    if not walls:
        return unconditioned(z, w)
    x, *tail = walls
    v = lambda p: 1 / (p - x) + 1 / x
    # Derivative of both transported dye coordinates and remaining contact sites.
    step = 0.0007 * min([abs(x), abs(z-x), abs(w-x), *[abs(y-x) for y in tail]]) ** 2
    def transported(a):
        return density(z+a*v(z), w+a*v(w), [y+a*v(y) for y in tail])
    derivative = (transported(-2*step) - 8*transported(-step)
                  + 8*transported(step) - transported(2*step)) / (12*step)
    weight = H / x**2 + sum(2 / (y-x)**2 for y in tail)
    return weight * density(z, w, tail) - derivative


def response(z, w, walls, gate):
    total, first, second, both = density(z, w, walls)
    if gate == "right":
        return both / first
    return (second - both) / (total - first)
