"""Laboratory-coordinate momentum integration for the simultaneous snapshot."""
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss

TRUE_PARAMETER = 1.07
SIGMA = .004


@lru_cache(maxsize=8)
def quadrature(order, angular):
    return (*leggauss(order), *leggauss(angular))


@lru_cache(maxsize=1024)
def energy(mass, temperature, gas_speed, analysis_speed, order=192, angular=128):
    z, w, mu, wm = quadrature(order, angular)
    boost = 1/np.sqrt(1-gas_speed*gas_speed)
    maximum_energy = boost*(1+abs(gas_speed))*(mass+60*temperature)
    maximum_rapidity = np.arccosh(maximum_energy/mass)
    q = (z+1)*maximum_rapidity/2
    E = mass*np.cosh(q)
    p = mass*np.sinh(q)
    scalar_density = np.exp(-(boost*(E[:,None]-gas_speed*p[:,None]*mu)-mass)/temperature)
    weight = w[:,None]*maximum_rapidity/2*p[:,None]**2*E[:,None]*wm*scalar_density
    measured_energy = (E[:,None]-analysis_speed*p[:,None]*mu)/np.sqrt(1-analysis_speed**2)
    return float(np.sum(weight*measured_energy)/np.sum(weight))


def predict(experiments, mass):
    return np.asarray([energy(mass,e['temperature'],e['gas_speed'],e['analysis_speed'])
                       for e in experiments])


def calibration_inputs():
    rows = [dict(temperature=T,gas_speed=b,analysis_speed=b)
            for T in [.2,.35,.55,.8] for b in [-.8,-.35,.35,.8]]
    rows += [dict(temperature=T,gas_speed=0.,analysis_speed=a)
             for T in [.25,.65] for a in [-.7,0.,.7]]
    return rows*12


def hidden_inputs():
    def rows(items):
        return [dict(temperature=T,gas_speed=b,analysis_speed=a) for T,b,a in items]
    return {
        'laboratory_energy': rows([(.45,.75,0),(.65,-.8,0),(.9,.9,0),
                                   (.55,-.85,0),(.8,.8,0),(.7,-.9,0)]),
        'opposed_analysis': rows([(.45,.75,-.25),(.65,-.8,.35),(.9,.9,-.5),
                                  (.55,-.85,.5),(.8,.8,-.35),(.7,-.9,.25)]),
        'moving_analysis': rows([(.45,.9,.3),(.65,-.9,-.4),(.9,.85,.25),
                                 (.55,-.85,-.2),(.8,.9,.45),(.7,-.8,-.25)]),
        'matched_anchors': rows([(.23,.7,.7),(.48,-.6,-.6),(.9,.9,.9),
                                 (.3,0,.8),(.6,0,-.8),(.95,0,0)])
    }
