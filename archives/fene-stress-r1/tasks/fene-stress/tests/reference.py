"""Independent Cartesian connector-force quadrature."""
from functools import lru_cache
import itertools
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import solve_continuous_lyapunov

TRUE_PARAMETER = 4.1


def cartesian(rate,temperature,drag,max_length,points=80):
    if max_length is None:
        drift=np.diag([rate,-rate])-2/drag*np.eye(2)
        c=solve_continuous_lyapunov(drift,-4*temperature/drag*np.eye(2))
        return float(c[0,0]-c[1,1]),c,0.
    q,w=leggauss(points)
    x=max_length*q[:,None]
    half=np.sqrt(max_length**2-x*x)
    y=half*q[None,:]
    weight=max_length*w[:,None]*half*w[None,:]
    gap=1-(x*x+y*y)/max_length**2
    potential=-max_length**2/2*np.log(gap)
    logdensity=(-potential+drag*rate*(x*x-y*y)/4)/temperature
    density=np.exp(logdensity-logdensity.max())
    weights=weight*density
    z=weights.sum()
    xx=float((weights*x*x).sum()/z);yy=float((weights*y*y).sum()/z)
    stress=float((weights*(x*x-y*y)/gap).sum()/z)
    # Pointwise stationary probability current, evaluated from the full force.
    dlogx=(-x/gap+drag*rate*x/2)/temperature
    dlogy=(-y/gap-drag*rate*y/2)/temperature
    currentx=rate*x-2*x/(drag*gap)-2*temperature/drag*dlogx
    currenty=-rate*y-2*y/(drag*gap)-2*temperature/drag*dlogy
    current=float(max(np.max(abs(currentx)),np.max(abs(currenty))))
    return stress,np.diag([xx,yy]),current



def experiment(rate, temperature, max_length):
    return dict(rate=rate, temperature=temperature, max_length=max_length)


def calibration_inputs():
    return [experiment(rate, temp, None) for rate, temp in
            itertools.product([-.22, -.15, -.08, .06, .14, .2], [.8, 1., 1.2])]*8


def hidden_inputs():
    return {
        'hookean_anchors': [experiment(.11, .9, None), experiment(-.19, 1.15, None)],
        'moderate_extension': [experiment(.4, .85, 2.55), experiment(.55, 1.15, 3.3),
                               experiment(.45, 1., 2.9), experiment(.6, .95, 3.4)],
        'strong_extension': [experiment(.75, 1.1, 2.6), experiment(.9, .8, 3.2),
                             experiment(.8, 1.2, 3.4), experiment(.95, 1., 2.8)],
        'reversed_extension': [experiment(-.4, .9, 3.4), experiment(-.7, 1.1, 2.5),
                               experiment(-.55, .8, 3.), experiment(-.85, 1.2, 2.75)],
    }


@lru_cache(maxsize=2048)
def response(rate, temperature, drag, max_length, points):
    return cartesian(rate, temperature, drag, max_length, points)[0]


def predict(experiments, drag=TRUE_PARAMETER, points=80):
    return np.array([response(e['rate'], e['temperature'], drag, e['max_length'], points) for e in experiments])
