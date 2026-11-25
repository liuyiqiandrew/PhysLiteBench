from functools import lru_cache
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar


@lru_cache(maxsize=1)
def single_species_coefficient():
    start = 1e-5
    def rhs(x,y):
        return [y[1],np.exp(-y[0])-2*y[1]/x]
    profile = solve_ivp(rhs,(start,12.),[start**2/6,start/3],
                        rtol=2e-11,atol=2e-13,dense_output=True).sol
    def condition(alpha):
        psi,gradient = profile(alpha)
        return alpha*np.exp(-psi)-gradient
    alpha = brentq(condition,5.,12.)
    return 1/(alpha*profile(alpha)[1])


def integrate(alpha,ratio,central):
    start = 1e-5
    def rhs(x,y):
        psi,gradient,light_mass,heavy_mass = y
        light = (1-central)*np.exp(-psi)
        heavy = central*np.exp(-ratio*psi)
        return [gradient,light+heavy-2*gradient/x,x*x*light,x*x*heavy]
    initial = [start**2/6,start/3,(1-central)*start**3/3,central*start**3/3]
    return solve_ivp(rhs,(start,alpha),initial,rtol=2e-10,atol=2e-12).y[:,-1]


def equilibrium(alpha,ratio,fraction):
    def composition(central):
        state = integrate(alpha,ratio,central)
        return state[3]/(state[2]+state[3])-fraction
    central = brentq(composition,0.,1.,xtol=1e-12)
    state = integrate(alpha,ratio,central)
    return alpha*state[1]


@lru_cache(maxsize=None)
def coefficient(ratio,fraction):
    if ratio == 1 or fraction == 0:
        return single_species_coefficient()
    # Continue at fixed global composition and find the FIRST eta maximum.
    grid = np.geomspace(.5,40.,40)
    previous = equilibrium(grid[0],ratio,fraction)
    middle = equilibrium(grid[1],ratio,fraction)
    for i in range(2,len(grid)):
        current = equilibrium(grid[i],ratio,fraction)
        if previous < middle and current < middle:
            fitted = minimize_scalar(lambda a:-equilibrium(a,ratio,fraction),
                                     bounds=(grid[i-2],grid[i]),method='bounded',
                                     options={'xatol':1e-7})
            return -1/fitted.fun
        previous,middle = middle,current
    raise RuntimeError('No first instability found in the continuation interval.')


def predict_at(experiments,gravity):
    return np.asarray([gravity*e['mass']/e['radius']*
                       coefficient(e['mass_ratio'],e['heavy_fraction']) for e in experiments])


class Model:
    def __init__(self):
        self.gravity = None

    def fit(self,records):
        inputs = [r['input'] for r in records]
        values = np.asarray([r['value'] for r in records])
        sigma = np.asarray([r['sigma'] for r in records])
        design = predict_at(inputs,1.)/sigma
        self.gravity = float(np.clip(design@(values/sigma)/(design@design),.7,1.4))
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.gravity)
