"""Physical phase-space dynamics and quadrature of observed field pairs."""
from functools import lru_cache
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import expm, solve_continuous_lyapunov

TRUE_PARAMETER = 1.07


@lru_cache(None)
def gaussian_rule(order):
    z,w = hermgauss(order)
    return np.sqrt(2)*z,w/np.sqrt(np.pi)


@lru_cache(maxsize=8192)
def response(stiffness,temperature,coupling,wavevector,delay,order=32):
    hessian = .6*stiffness*np.eye(5)
    for j in range(4):
        link = np.zeros(5);link[j]=-1;link[j+1]=1
        hessian += stiffness*coupling*np.outer(link,link)
    drift = np.block([[np.zeros((5,5)),np.eye(5)],[-hessian,-.4*np.eye(5)]])
    noise = np.zeros((10,10));noise[5:,5:] = .8*temperature*np.eye(5)
    stationary = solve_continuous_lyapunov(drift,-noise)
    cross = (expm(delay*drift)@stationary)[:5,:5]
    z,w = gaussian_rule(order);total=0.
    for i in range(5):
        for j in range(5):
            covariance = np.array([[stationary[i,i],cross[i,j]],[cross[i,j],stationary[j,j]]])
            values,vectors = np.linalg.eigh(covariance)
            if min(values)<-1e-10:
                raise ValueError('Invalid joint stationary covariance.')
            transform = vectors*np.sqrt(np.maximum(values,0.))
            first = transform[0,0]*z[:,None]+transform[0,1]*z[None,:]
            second = transform[1,0]*z[:,None]+transform[1,1]*z[None,:]
            total += w@np.cos(wavevector*(i-j+first-second))@w/5
    return float(total)


def predict(experiments,stiffness=TRUE_PARAMETER,order=32):
    return np.asarray([response(float(stiffness),float(e['temperature']),float(e['coupling']),
                                float(e['wavevector']),float(e['delay']),order) for e in experiments])


def calibration_inputs():
    return [{'temperature':t,'coupling':c,'wavevector':2*np.pi,'delay':d}
            for t in [.04,.08,.12] for c,d in [(1.,0.),(0.,.3),(0.,.8)]]


def hidden_inputs():
    groups={}
    for temperature in [.04,.08,.12]:
        groups[f'temperature_{temperature}']=[{'temperature':temperature,'coupling':c,'wavevector':q,'delay':d}
            for c in [.8,1.] for q in [5.9,2*np.pi] for d in [1.,1.25,1.5]]
    groups['exact_anchor']=[{'temperature':.06,'coupling':c,'wavevector':q,'delay':d}
                           for c,q,d in [(1.,2.2,0.),(.4,4.1,0.),(0.,1.4,.7),(0.,5.1,2.3)]]
    return groups
