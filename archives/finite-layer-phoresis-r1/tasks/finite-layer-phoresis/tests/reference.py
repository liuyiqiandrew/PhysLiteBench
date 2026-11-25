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
def flow(radius, width, strength, cells=256):
    if strength == 0:
        return 0.
    end = radius+width
    c = concentration(radius,width,strength,cells)
    def rhs(r,y):
        forcing = 2*strength*(end-r)*c(r)/width**2
        return np.array([y[1],y[2],y[3],forcing+4*y[2]/r**2-8*y[1]/r**3+8*y[0]/r**4])
    def bc(left,right):
        return np.array([left[0],left[1],right[2]-2*right[0]/end**2,
                         right[3]-2*right[1]/end**2+4*right[0]/end**3])
    grid = np.linspace(radius,end,cells+1)
    solution = solve_bvp(rhs,bc,grid,np.zeros((4,cells+1)),tol=5e-9,max_nodes=20000)
    if not solution.success:
        raise RuntimeError(solution.message)
    boundary = solution.sol(end)
    return -2/(3*end)*(boundary[1]+boundary[0]/end)


@lru_cache(128)
def unit_response(kind,width,strength,radius=1.,cells=256):
    if kind == 'wall':
        x,w=leggauss(64)
        z=width*(x+1)/2
        return -width/2*np.sum(w*z*np.expm1(-strength*(1-z/width)**2))
    return (4*flow(radius,width,strength,2*cells)-flow(radius,width,strength,cells))/3


def predict(experiments,viscosity=TRUE_PARAMETER,cells=256):
    return np.array([unit_response(e['kind'],e['width'],e['strength'],e.get('radius',1.),cells)
                     for e in experiments])/viscosity


def experiment(kind='sphere',radius=1.,width=2.2,strength=1.2):
    out=dict(kind=kind,width=float(width),strength=float(strength))
    if kind=='sphere':out['radius']=float(radius)
    return out


def calibration_inputs():
    unique=[experiment('wall',width=w,strength=s) for w in [.3,.65,1.] for s in [-2.,-1.,-.5,.5,1.,2.]]
    return [dict(e) for _ in range(16) for e in unique]


def hidden_inputs():
    return {
        'repulsive_layers':[experiment(radius=a,width=a*d,strength=s) for a,d,s in [(.7,2.2,.7),(1.,2.5,1.2),(1.3,2.4,2.),(.9,2.,1.7)]],
        'attractive_layers':[experiment(radius=a,width=a*d,strength=s) for a,d,s in [(.8,2.4,-.6),(1.1,2.3,-1.1),(1.3,2.5,-1.8),(.95,2.1,-2.)]],
        'geometry_sweep':[experiment(radius=a,width=w,strength=s) for a,w,s in [(.7,1.5,1.5),(1.,2.3,-1.5),(1.25,3.4,.9),(1.3,2.7,-.9)]],
        'wall_anchors':[experiment('wall',width=w,strength=s) for w,s in [(.4,.8),(.75,-1.3),(1.2,1.7),(.8,-.7)]]
    }
