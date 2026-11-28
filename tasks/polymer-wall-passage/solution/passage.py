from scipy.special import hyp2f1
import numpy as np


def bulk(z, w):
    # Unbounded bubble measure for enclosing the first dye and for enclosing both.
    a, b = -(1/z).imag, -(1/w).imag
    s = abs((z-w)/(z-w.conjugate()))**2
    g = 1 - s*hyp2f1(1., 4/3, 5/3, 1-s)
    return np.array([a*a, a*b*g])/4


def contacts(z, w, walls):
    if not walls:
        return bulk(z,w)
    x, *tail = walls
    def v(p):
        return 1/(p-x)+1/x
    h = .0006*min(abs(x),abs(z-x),abs(w-x),*[abs(x-y) for y in tail])**2
    def moved(a):
        return contacts(z+a*v(z),w+a*v(w),[y+a*v(y) for y in tail])
    derivative = (moved(-2*h)-8*moved(-h)+8*moved(h)-moved(2*h))/(12*h)
    weight = 2/x**2 + sum(2/(y-x)**2 for y in tail)
    return weight*contacts(z,w,tail)-derivative


def cumulative(radius, z, w, walls):
    def disk(p):
        return p/(1+(p/radius)**2)
    jac = [((1-(x/radius)**2)/(1+(x/radius)**2)**2)**2 for x in walls]
    return np.prod(jac)*contacts(disk(z),disk(w),[disk(x) for x in walls])


def response(radius, z, w, walls):
    h=.0015*radius
    def measure(r):
        return cumulative(r,z,w,walls)
    one,both=(measure(radius-2*h)-8*measure(radius-h)+8*measure(radius+h)-measure(radius+2*h))/(12*h)
    return both/one
