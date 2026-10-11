from functools import lru_cache
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


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


def coefficient(ratio,fraction):
    mean_mass = 1/(1-fraction+fraction/ratio)
    return mean_mass*single_species_coefficient()


def predict_at(experiments,gravity):
    return np.asarray([gravity*e['mass']/e['radius']*
                       coefficient(e['mass_ratio'],e['heavy_fraction']) for e in experiments])


class Model:
    def __init__(self):
        self.gravity = None

    def fit(self,records):
        raise NotImplementedError('Fit gravitational strength from calibration.')

    def predict(self,experiments):
        return predict_at(experiments,self.gravity)
