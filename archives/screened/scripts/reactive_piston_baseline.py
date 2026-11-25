from functools import lru_cache
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

EQUILIBRIUM = .75


@lru_cache(maxsize=128)
def trajectory(initial, left_fraction):
    volume_initial=np.array([left_fraction, 1-left_fraction])
    amounts=np.array(initial).reshape(2, 2)*volume_initial[:, None]
    def concentrations(extent):
        amount=amounts+extent[:, None]*np.array([-1., 2.])
        volume=volume_initial
        return amount/volume[:, None],volume
    def rhs(time, extent):
        c,volume=concentrations(extent)
        return volume*(c[:, 0]-c[:, 1]**2/EQUILIBRIUM)
    sol=solve_ivp(rhs,(0.,24.),np.zeros(2),method='DOP853',max_step=.1,
                  rtol=2e-11,atol=2e-13,dense_output=True).sol
    return sol,concentrations


class Model:
    def __init__(self):
        self.rate=None

    def fit(self, records):
        experiments=[r['input'] for r in records]
        values=np.array([r['value'] for r in records])
        sigma=np.array([r['sigma'] for r in records])
        def residual(parameter):
            self.rate=float(parameter[0])
            return (self.predict(experiments)-values)/sigma
        answer=least_squares(residual,[.1],bounds=(.03,.3),
                             ftol=1e-12,xtol=1e-12,gtol=1e-12)
        self.rate=float(answer.x[0])
        return self

    def predict(self, experiments):
        out=[]
        for e in experiments:
            sol,concentrations=trajectory(tuple(np.asarray(e['initial'],dtype=float).ravel()),e['left_fraction'])
            c,volume=concentrations(sol(self.rate*e['time']))
            out.append(c[e['chamber'],1])
        return np.array(out)
