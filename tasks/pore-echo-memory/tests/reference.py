"""First-passage resolvent on a spatial finite-difference mesh."""
import math

import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh_tridiagonal

TRUE_PARAMETER = .873


def tail_discrete(drift, loss, order):
    # Generator d²/dx² - 2*drift*d/dx on [0,1], reflection at 0, exit at 1.
    n = order
    up, down = n*n-drift*n, n*n+drift*n
    birth = np.full(n, up, dtype=float)
    birth[0] = 2*n*n
    eig, modes = eigh_tridiagonal(np.full(n,-2.*n*n),np.sqrt(birth[:-1]*down))
    pi_end = math.prod(birth[:-1]/down)
    shifted = loss-eig
    weight = up*math.sqrt(pi_end)*modes[0]*modes[-1]/shifted
    def tail(t):
        return weight @ np.exp(-shifted*max(0,t))
    return tail


def predict(q, intensity=TRUE_PARAMETER, order=160):
    unit = q["u"]**2
    r, force = intensity*q["a"]*unit, q["u"]*q["f"]
    s = q["y"]/unit
    a, b = tail_discrete(force,r,order), tail_discrete(-force,r,order)
    count = r/(1-a(0)*b(0))
    lo, hi = (-np.inf,np.inf) if q["window"] is None else np.array(q["window"])/unit
    points = [lo]+[z for z in sorted({0.,s}) if lo<z<hi]+[hi]
    return count*sum(quad(lambda z: a(z)*b(s-z), l,h,epsabs=1e-9)[0] for l,h in zip(points,points[1:]))
