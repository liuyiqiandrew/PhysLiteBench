"""Disorder-averaged transfer recursion, independent of spectral inversion."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import roots_legendre

TRUE_PARAMETER = .83


def experiment(count=1, wave_number=1.):
    return dict(count=count, wave_number=wave_number)


def calibration_inputs():
    return [experiment(n,k) for n in [1,2] for k in [.8,.9,1.,1.1,1.2,1.3,1.4,1.6]]


def hidden_inputs():
    return {
        'shared_anchors': [experiment(n,k) for n in [0,1,2] for k in [.85,1.15,1.55]],
        'short_chains': [experiment(n,k) for n in [3,4,5] for k in [.8,.9,1.1]],
        'long_chains': [experiment(n,k) for n in [8,10,12] for k in [.8,1.,1.2]],
        'weak_barriers': [experiment(n,k) for n in [6,9,12] for k in [1.2,1.4,1.6]],
    }


def coherent_mean(transmission, count, grid=1200, angles=96):
    if count == 0:
        return 1.
    if count == 1:
        return transmission
    if count == 2:
        return transmission/(2-transmission)
    jump=np.arccosh(2/transmission-1)
    nodes,weights=roots_legendre(angles)
    cosine=np.cos(np.pi*(nodes+1)/2)
    x=np.linspace(0,count*jump,grid+1)
    values=1/np.cosh(x/2)**2
    for i in range(1,count):
        spline=CubicSpline(x,values,bc_type=((1,0.),'not-a-knot'))
        x=np.linspace(0,(count-i)*jump,grid+1)
        argument=np.cosh(x[:,None])*np.cosh(jump)+np.sinh(x[:,None])*np.sinh(jump)*cosine
        following=np.arccosh(np.maximum(argument,1))
        values=spline(following)@(weights/2)
    return float(values[-1])



def predict(experiments, strength, grid=1200, angles=96):
    return np.asarray([coherent_mean(1/(1+(strength/e['wave_number'])**2),
                                     e['count'], grid, angles) for e in experiments])


def transfer_samples(strength, wave_number, count, samples=131072, seed=613017):
    """Direct delta-jump transfer matrices with actual fixed random gaps."""
    rng = np.random.default_rng(seed)
    z = strength/wave_number
    barrier = np.array([[1-1j*z,-1j*z],[1j*z,1+1j*z]])
    matrix = np.broadcast_to(np.eye(2,dtype=complex),(samples,2,2)).copy()
    for j in range(count):
        matrix = barrier@matrix
        if j+1 < count:
            phase = np.exp(1j*rng.uniform(0,2*np.pi,samples))
            matrix[:,0,:] *= phase[:,None]
            matrix[:,1,:] *= phase.conj()[:,None]
    return 1/np.abs(matrix[:,1,1])**2
