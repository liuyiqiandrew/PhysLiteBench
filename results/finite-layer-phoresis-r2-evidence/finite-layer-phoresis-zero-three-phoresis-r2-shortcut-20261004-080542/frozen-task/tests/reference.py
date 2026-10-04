"""Conservative solute finite elements and a force-free Stokes streamfunction."""
from functools import lru_cache
import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_banded
from numpy.polynomial.legendre import leggauss

TRUE_PARAMETER = 1.2


def concentration(radius, width, strength, cells):
    grid = np.linspace(radius, radius+width, cells+1)
    step = width/cells
    diagonal = np.zeros(cells+1)
    upper = np.zeros(cells)
    nodes, weights = leggauss(4)
    for xi, weight in zip(nodes, weights):
        N = np.array([(1-xi)/2, (1+xi)/2])
        r = grid[:-1]+step*(xi+1)/2
        boltzmann = np.exp(-strength*((radius+width-r)/width)**2)
        gradient = r*r*boltzmann/step**2
        factor = weight*step/2
        diagonal[:-1] += factor*(gradient+2*boltzmann*N[0]**2)
        diagonal[1:] += factor*(gradient+2*boltzmann*N[1]**2)
        upper += factor*(-gradient+2*boltzmann*N[0]*N[1])
    end = grid[-1]
    diagonal[-1] += 2*end
    rhs = np.zeros(cells+1)
    rhs[-1] = 3*end**2
    bands = np.zeros((3,cells+1))
    bands[0,1:] = upper
    bands[1] = diagonal
    bands[2,:-1] = upper
    h = solve_banded((1,1), bands, rhs)
    profile = CubicSpline(grid,h)
    return lambda r: profile(r)*np.exp(-strength*((end-r)/width)**2)


@lru_cache(256)
def flow(radius, width, strength, slip, cells=256):
    if strength == 0:
        return 0.
    end = radius+width
    c = concentration(radius,width,strength,cells)
    def rhs(r,y):
        forcing = 2*strength*(end-r)*c(r)/width**2
        return np.array([y[1],y[2],y[3],forcing+4*y[2]/r**2-8*y[1]/r**3+8*y[0]/r**4])
    def bc(left,right):
        return np.array([left[0],(1+2*slip/radius)*left[1]-slip*left[2],right[2]-2*right[0]/end**2,
                         right[3]-2*right[1]/end**2+4*right[0]/end**3])
    grid = np.linspace(radius,end,cells+1)
    solution = solve_bvp(rhs,bc,grid,np.zeros((4,cells+1)),tol=5e-9,max_nodes=20000)
    if not solution.success:
        raise RuntimeError(solution.message)
    boundary = solution.sol(end)
    return -2/(3*end)*(boundary[1]+boundary[0]/end)


@lru_cache(256)
def unit_response(radius,width,strength,slip,cells=256):
    return (4*flow(radius,width,strength,slip,2*cells)-flow(radius,width,strength,slip,cells))/3


def predict(experiments,viscosity=TRUE_PARAMETER,cells=256):
    return np.array([unit_response(e['radius'],e['width'],e['strength'],e['slip'],cells)
                     for e in experiments])/viscosity


def experiment(radius=1.,width=.7,strength=1.2,slip=.5):
    return dict(radius=float(radius),width=float(width),strength=float(strength),slip=float(slip))


def calibration_inputs():
    unique=[experiment(a,w,s,0.) for a in [.75,1.2] for w in [.35,1.,2.] for s in [-1.5,-.6,.6,1.5]]
    return [dict(e) for _ in range(6) for e in unique]


def hidden_inputs():
    return {
        'repulsive_layers':[experiment(a,w,s,b) for a,w,s,b in [(.7,.3,.7,.4),(1.,.7,1.2,.8),(1.3,1.3,2.,1.6),(.9,1.8,1.7,1.2)]],
        'attractive_layers':[experiment(a,w,s,b) for a,w,s,b in [(.8,.3,-.6,.6),(1.1,.8,-1.1,.7),(1.3,1.5,-1.8,1.3),(.95,2.1,-2.,1.4)]],
        'geometry_sweep':[experiment(a,w,s,b) for a,w,s,b in [(.7,.35,1.5,.7),(1.,1.1,-1.5,1.4),(1.25,2.5,.9,1.8),(1.3,.6,-.9,.4)]],
        'no_slip_anchors':[experiment(a,w,s,0.) for a,w,s in [(.8,.4,.8),(1.1,.75,-1.3),(1.2,1.8,1.7),(1.,2.5,-.7)]]
    }
