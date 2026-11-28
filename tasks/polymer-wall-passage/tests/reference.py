"""Author reference using the finite-notch restriction map and an ODE for the angular law."""
import numpy as np
from scipy.integrate import solve_ivp

# The regular solution at s=1 has series 1-s - 4/5 (1-s) + ...
def start(t):
    term, total = 1., 1.
    for n in range(1, 14):
        term *= (4/3+n-1)/(5/3+n-1)*t
        total += term
    return 1-(1-t)*total


shape_solution = solve_ivp(
    lambda s, f: [(s-1+(s+1)*f[0])/(3*s*(1-s))],
    (0.98, 1.e-5), [start(.02)], method="DOP853", dense_output=True,
    rtol=2.e-13, atol=5.e-15)


def angular(z, w):
    c, d = np.angle(z), np.angle(w)
    u, v = np.cos(c/2)**2, np.cos(d/2)**2
    s = abs(z-w)**2 / abs(z-w.conjugate())**2
    assert 1.e-5 < s <= 1.
    g = start(1-s) if s >= .98 else float(shape_solution.sol(s)[0])
    return np.array([1., u, v, u*v+np.sin(c)*np.sin(d)*g/4])


def remove(p, x, a):
    # Principal square root of the quotient fixes the hydrodynamic branch.
    def g(q):
        return x+(q-x)*np.sqrt(1.+2.*a/(q-x)**2)
    def jac(q):
        return 1./np.sqrt(1.+2.*a/(q-x)**2)
    z, w, *others = p
    return [g(q)-g(0.) for q in (z,w,*others)], jac(0.)**(5/8)*np.prod(
        [jac(y)**2 for y in others])


def finite(z, w, walls, step):
    if not walls:
        return angular(z, w)
    x, *others = walls
    a = step*min(abs(x), abs(z-x), abs(w-x), *[abs(x-y) for y in others])**2
    moved, factor = remove([z,w,*others], x, a)
    mz, mw, *tail = moved
    return (finite(z,w,others,step) - factor*finite(mz,mw,tail,step))/a


def extrapolate(z, w, walls, step=.003):
    limits = [finite(z,w,walls,step/2**j) for j in range(3)]
    return (limits[0]-6*limits[1]+8*limits[2])/3


def value(r, aspect):
    z, w = [complex(x,aspect*y) for x,y in r["dyes"]]
    total, first, second, both = extrapolate(z,w,list(reversed(r["walls"])))
    if r["gate"]=="off":
        return second
    return both/first if r["gate"]=="right" else (second-both)/(total-first)
