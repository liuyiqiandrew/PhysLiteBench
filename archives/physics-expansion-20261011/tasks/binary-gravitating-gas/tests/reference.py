"""Two-species canonical free-energy Hessian at fixed global composition."""
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import legvander,legder,legval
from scipy.integrate import solve_ivp
from scipy.linalg import eigh,block_diag
from scipy.optimize import brentq
from scipy.special import roots_legendre

TRUE_PARAMETER = 1.07


def experiment(mass=1.,radius=1.,ratio=1.,fraction=0.):
    return dict(mass=mass,radius=radius,mass_ratio=ratio,heavy_fraction=fraction)


def calibration_inputs():
    scales = [(.85,.8),(1.,1.4),(1.15,1.1),(1.3,.9),(1.45,1.3),(1.6,1.5)]
    return [experiment(m,r,ratio,fraction) for m,r in scales
            for ratio,fraction in [(1.,.1),(1.,.5),(2.,0.),(3.5,0.),(5.,0.)]]


def hidden_inputs():
    scales = [(.9,1.35),(1.2,.75),(1.55,1.15)]
    return {
        'shared_anchors': [experiment(m,r,ratio,q) for m,r in scales
                           for ratio,q in [(1.,.3),(4.,0.)]],
        'moderate_masses': [experiment(m,r,2.,q) for m,r in scales for q in [.2,.3,.5]],
        'unequal_masses': [experiment(m,r,3.,q) for m,r in scales for q in [.1,.25,.5]],
        'heavy_component': [experiment(m,r,5.,q) for m,r in scales for q in [.1,.3,.5]],
    }


def integrate(alpha,ratio,central):
    # Independent equilibrium integration in logarithmic radius.
    start = 1e-5
    def rhs(log_x,state):
        x = np.exp(log_x)
        psi,eta,light_mass,heavy_mass = state
        light = (1-central)*np.exp(-psi)
        heavy = central*np.exp(-ratio*psi)
        return [eta,x*x*(light+heavy)-eta,x**3*light,x**3*heavy]
    initial = [start**2/6,start**2/3,(1-central)*start**3/3,central*start**3/3]
    return solve_ivp(rhs,(np.log(start),np.log(alpha)),initial,
                     rtol=2e-11,atol=2e-13,max_step=.08,dense_output=True)


@lru_cache(maxsize=2048)
def equilibrium(alpha,ratio,fraction):
    if fraction == 0:
        central = 0.
    elif ratio == 1:
        central = fraction
    else:
        def composition(central):
            state = integrate(alpha,ratio,central).y[:,-1]
            return state[3]/(state[2]+state[3])-fraction
        central = brentq(composition,0.,1.,xtol=1e-12)
    profile = integrate(alpha,ratio,central)
    return profile.y[1,-1],central,profile


@lru_cache(maxsize=None)
def basis(modes,points):
    z,weight = roots_legendre(points)
    r,weight = (z+1)/2,weight/2
    polynomial = legvander(2*r*r-1,modes-1).T
    derivative = np.asarray([legval(2*r*r-1,legder(np.eye(modes)[j]))
                             for j in range(modes)])*4*r
    shape = r**3*(1-r*r)*polynomial
    slope = (3*r*r-5*r**4)*polynomial+r**3*(1-r*r)*derivative
    return r,weight,shape,slope


def curvature(alpha,ratio,fraction,modes=64,points=256):
    eta,central,profile = equilibrium(alpha,ratio,fraction)
    theta = 1/eta
    r,weight,shape,slope = basis(modes,points)
    psi = profile.sol(np.log(alpha*r))[0]
    light = alpha**2*theta*(1-central)*np.exp(-psi)/(4*np.pi)
    a_light = (slope*(weight/(4*np.pi*light*r*r)))@slope.T
    gravity = (shape*(weight/(r*r)))@shape.T
    if fraction == 0:
        entropy = a_light
        matrix = entropy-gravity/theta
    else:
        heavy = alpha**2*theta*central*np.exp(-ratio*psi)/(4*np.pi)
        a_heavy = (slope*(weight/(4*np.pi*ratio*heavy*r*r)))@slope.T
        entropy = block_diag(a_light,a_heavy)
        matrix = entropy-np.block([[gravity,gravity],[gravity,gravity]])/theta
    return float(eigh(matrix,entropy,subset_by_index=[0,0],check_finite=False)[0][0])


@lru_cache(maxsize=None)
def critical(ratio,fraction,modes=64,points=256):
    if ratio == 1 or fraction == 0:
        ratio,fraction = 1.,0.
    grid = np.geomspace(.5,40.,32)
    for left,right in zip(grid[:-1],grid[1:]):
        if curvature(right,ratio,fraction,modes,points) < 0:
            alpha = brentq(lambda a:curvature(a,ratio,fraction,modes,points),
                           left,right,xtol=1e-8)
            return 1/equilibrium(alpha,ratio,fraction)[0],alpha
    raise RuntimeError('No first free-energy instability found.')


def predict(experiments,gravity,modes=64,points=256):
    return np.asarray([gravity*e['mass']/e['radius']*
                       critical(e['mass_ratio'],e['heavy_fraction'],modes,points)[0]
                       for e in experiments])
