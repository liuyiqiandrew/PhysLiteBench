import itertools
import numpy as np
from scipy.linalg import expm_frechet

TRUE_PARAMETER=1.1
SX=np.array([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]])/np.sqrt(2.)
SZ=np.diag([1.,0.,-1.])
Q=SZ@SZ


def experiment(anisotropy=1.,transverse=1.,longitudinal=.25,temperature=.35):
    return dict(anisotropy=anisotropy,transverse=transverse,longitudinal=longitudinal,temperature=temperature)


def response(e,coupling):
    h=e['anisotropy']*Q+e['transverse']*SX+e['longitudinal']*SZ
    thermal,change=expm_frechet(-h/e['temperature'],coupling*Q/e['temperature'])
    z=np.trace(thermal)
    force_numerator=coupling*np.trace(Q@thermal)
    numerator_change=coupling*np.trace(Q@change)
    return float((numerator_change*z-force_numerator*np.trace(change))/z**2)


def predict(experiments,coupling=TRUE_PARAMETER):
    return np.array([response(e,coupling) for e in experiments])


def calibration_inputs():
    return [experiment(d,0.,z,t) for repeat in range(6) for d,z,t in itertools.product([.7,1.,1.3],[-.6,0.,.6],[.2,.35,.55,.8])]


def hidden_inputs():
    return {
        'transverse': [experiment(1.,h,.25,.4) for h in [.65,.9,1.2,1.5]],
        'temperature': [experiment(.9,1.1,-.35,t) for t in [.22,.3,.4,.5]],
        'anisotropy_bias': [experiment(d,h,z,t) for d,h,z,t in [(.75,.7,-.65,.3),(1.25,1.4,.6,.5),(1.1,.85,.1,.35),(.85,1.3,-.4,.45)]],
        'commuting_anchors': [experiment(.85,0.,.25,.4),experiment(1.15,0.,-.45,.7)]
    }
