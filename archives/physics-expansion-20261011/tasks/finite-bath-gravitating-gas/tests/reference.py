"""Radial constrained-entropy Hessian; no turning-point stability formula."""
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import legvander, legder, legval
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.special import roots_legendre

TRUE_PARAMETER = 1.08


def experiment(observable='critical_temperature', mass=1., radius=1., bath=0., contrast=None):
    row = dict(observable=observable, mass=mass, radius=radius, bath_capacity=bath)
    if contrast is not None:
        row['contrast'] = contrast
    return row


def calibration_inputs():
    scales = [(.85,.8),(1.,1.4),(1.15,1.1),(1.3,.9),(1.45,1.3),(1.6,1.5)]
    rows = [experiment('temperature',m,r,b,c) for m,r in scales
            for b,c in [(0.,2.),(.3,5.),(.8,10.),(1.5,20.)]]
    return rows+[experiment(mass=m,radius=r,bath='infinite') for m,r in scales]


def hidden_inputs():
    scales = [(.9,1.35),(1.2,.75),(1.55,1.15)]
    return {
        'shared_anchors': [experiment('temperature',m,r,b,c) for m,r in scales
                           for b,c in [(2.,1.5),('infinite',25.)]]+
                          [experiment(mass=m,radius=r,bath='infinite') for m,r in scales],
        'small_reservoir': [experiment(mass=m,radius=r,bath=b) for m,r in scales for b in [0.,.1,.3]],
        'medium_reservoir': [experiment(mass=m,radius=r,bath=b) for m,r in scales for b in [.4,.7,.9]],
        'larger_reservoir': [experiment(mass=m,radius=r,bath=b) for m,r in scales for b in [1.,1.2,1.5]],
    }


@lru_cache(maxsize=1)
def profile():
    # Integrate equilibrium in logarithmic radius, independently of oracle IVP.
    x0 = 1e-5
    def rhs(log_x, state):
        u, v, psi = state
        return [u*(3-u-v), v*(u-1), v]
    return solve_ivp(rhs, (np.log(x0), np.log(60.)),
                     [3-x0*x0/5, x0*x0/3, x0*x0/6],
                     rtol=2e-12, atol=2e-14, max_step=.02, dense_output=True).sol


def equilibrium(alpha):
    return profile()(np.log(alpha))


@lru_cache(maxsize=None)
def basis(modes, points):
    z, weight = roots_legendre(points)
    r, weight = (z+1)/2, weight/2
    polynomial = legvander(2*r*r-1,modes-1).T
    derivative = np.asarray([legval(2*r*r-1,legder(np.eye(modes)[j]))
                             for j in range(modes)])*4*r
    shape = r**3*(1-r*r)*polynomial
    slope = (3*r*r-5*r**4)*polynomial+r**3*(1-r*r)*derivative
    return r, weight, shape, slope


def curvature(alpha, bath, modes=96, points=256):
    # q(r)=delta M(r); q(0)=q(1)=0, q=O(r^3). G=M=R=1.
    theta = 1/equilibrium(alpha)[1]
    r, weight, shape, slope = basis(modes,points)
    u, v, psi = equilibrium(alpha*r)
    density = alpha**2*theta*np.exp(-psi)/(4*np.pi)
    enclosed_mass = r*theta*v
    entropy = (slope*(weight/(4*np.pi*density*r*r)))@slope.T
    gravity = (shape*(weight/(r*r)))@shape.T
    energy = shape@(weight*enclosed_mass/(r*r))
    matrix = entropy-gravity/theta
    if bath != 'infinite':
        matrix += 2*np.outer(energy,energy)/((3+2*bath)*theta**2)
    return float(eigh(matrix,entropy,subset_by_index=[0,0],check_finite=False)[0][0])


@lru_cache(maxsize=None)
def critical(bath, modes=96, points=256):
    alpha = brentq(lambda x: curvature(x,bath,modes,points),6.,45.,xtol=1e-8)
    return 1/equilibrium(alpha)[1], alpha


@lru_cache(maxsize=None)
def ordinary(contrast):
    alpha = brentq(lambda x: equilibrium(x)[2]-np.log(contrast),.01,9.)
    return 1/equilibrium(alpha)[1]


def predict(experiments, gravity, modes=96, points=256):
    return np.asarray([gravity*e['mass']/e['radius']*
                       (ordinary(e['contrast']) if e['observable']=='temperature'
                        else critical(e['bath_capacity'],modes,points)[0])
                       for e in experiments])
