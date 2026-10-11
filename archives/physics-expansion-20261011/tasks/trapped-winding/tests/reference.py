"""Stationary two-radius heat-kernel integration for continuous winding."""
from functools import lru_cache
import numpy as np
from scipy.special import iv, roots_genlaguerre

TRUE_PARAMETER = 1.09


def experiment(k=1., temperature=1., probe=1.):
    return dict(stiffness=k,temperature=temperature,probe=probe)


def calibration_inputs():
    return [experiment(k,t,p) for k in [.6,.9,1.2,1.6]
            for t in [.5,1.,1.8] for p in [-1.,1.]]


def hidden_inputs():
    return {
        'shared_anchors': [experiment(k,t,p) for k,t in [(.7,.8),(1.1,1.3),(1.7,1.7)] for p in [-1.,0.,1.]],
        'fractional_probes': [experiment(k,t,p) for k,t in [(.8,.5),(1.3,1.1),(1.8,1.8)] for p in [.25,.5,.75]],
        'higher_probes': [experiment(k,t,p) for k,t in [(.6,1.5),(1.,.7),(1.6,1.2)] for p in [1.4,1.8,2.4]],
        'negative_probes': [experiment(k,t,p) for k,t in [(.9,1.6),(1.2,.6),(1.7,1.4)] for p in [-.35,-.65,-1.6]],
    }


@lru_cache(maxsize=256)
def log_scaled_characteristic(probe, time, order=64):
    """Log of C_lambda(t)/rho^abs(lambda), rho=exp(-mu*k*t).

    Integrate the I_abs(lambda) angular heat kernel over both stationary
    radii. Dimensionless squared radii x,y have exp(-x-y) weights.
    Generalized Laguerre absorbs their x^(nu/2) y^(nu/2) origin factors.
    """
    nu = abs(probe)
    rho = np.exp(-time)
    x, weights = roots_genlaguerre(order,nu/2)
    product = rho*np.sqrt(x[:,None]*x[None,:])
    kernel = iv(nu,2*product)/product**nu
    integral = (1-rho*rho)*float(weights@kernel@weights)
    return float(np.log(integral))


def predict(experiments, mobility, times=(10.,14.), order=64):
    result = []
    for e in experiments:
        probe = e['probe']
        if probe == 0:
            result.append(0.)
            continue
        t1,t2 = times
        log1 = -abs(probe)*t1+log_scaled_characteristic(probe,t1,order)
        log2 = -abs(probe)*t2+log_scaled_characteristic(probe,t2,order)
        result.append(-mobility*e['stiffness']*(log2-log1)/(t2-t1))
    return np.asarray(result)
