"""Free-energy derivative with independent surface-impedance reflection."""
import numpy as np
from functools import lru_cache
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
TRUE_PARAMETER=6.5

@lru_cache(None)
def grid(order):
    x,w=leggauss(order)
    return (x+1)/2,w/2


def boundary_reflection(q,xi,plasma,relaxation):
    # Electromagnetic impedances at imaginary frequency; squared amplitudes are basis-independent.
    inverse_epsilon=xi*(xi+relaxation)/(xi*(xi+relaxation)+plasma*plasma)
    decay=np.sqrt(q*q+plasma*plasma*xi/(xi+relaxation))
    electric=(q-decay)/(q+decay)
    magnetic=(decay*inverse_epsilon-q)/(decay*inverse_epsilon+q)
    return electric*electric,magnetic*magnetic


def free_energy(e,plasma,order=128,static_frequency=1e-15):
    a=e['separation'];t=e['temperature'];gamma=e['relaxation']
    z,w=grid(order)
    if t==0:
        y=44*z[:,None];q=y/(2*a);xi=q*z[None,:]**2
        re,rm=boundary_reflection(q,xi,plasma,gamma)
        terms=np.log1p(-re*np.exp(-y))+np.log1p(-rm*np.exp(-y))
        return np.sum(y*y*terms*(44*w[:,None])*(2*z[None,:]*w[None,:]))/(32*np.pi*np.pi*a**3)
    spacing=4*np.pi*a*t
    frequencies=np.arange(1,max(2,int(np.ceil(44/spacing))+1))
    y=spacing*frequencies[:,None]+44*z[None,:];q=y/(2*a)
    xi=np.maximum(2*np.pi*t*frequencies[:,None],static_frequency)
    re,rm=boundary_reflection(q,xi,plasma,gamma)
    terms=np.log1p(-re*np.exp(-y))+np.log1p(-rm*np.exp(-y))
    positive=np.sum(y*terms*(44*w[None,:]))
    def static_integrand(y):
        re,rm=boundary_reflection(y/(2*a),static_frequency,plasma,gamma)
        return y*(np.log1p(-re*np.exp(-y))+np.log1p(-rm*np.exp(-y)))
    static=quad(static_integrand,0.,44.,epsabs=1e-13,epsrel=1e-13)[0]
    return t*(positive+.5*static)/(8*np.pi*a*a)


def pressure(e,plasma,order=128,step=1e-4):
    h=step*e['separation']
    fm2=free_energy(dict(e,separation=e['separation']-2*h),plasma,order)
    fm1=free_energy(dict(e,separation=e['separation']-h),plasma,order)
    fp1=free_energy(dict(e,separation=e['separation']+h),plasma,order)
    fp2=free_energy(dict(e,separation=e['separation']+2*h),plasma,order)
    return -(fm2-8*fm1+8*fp1-fp2)/(12*h)


def predict(experiments,plasma):return np.array([pressure(e,plasma) for e in experiments])


def experiment(a,t,gamma):return dict(separation=a,temperature=t,relaxation=gamma)


def calibration_inputs():
    return [experiment(a,0.,gamma) for repeat in range(16) for a in [.55,.8,1.15,1.7] for gamma in [.18,.4,.75]]


def hidden_inputs():
    return {
      'thermal':[experiment(a,t,g) for a,t,g in [(1.1,.35,.18),(1.4,.5,.4),(1.8,.65,.75),(.8,.7,.3)]],
      'mixed_scales':[experiment(a,t,g) for a,t,g in [(.65,.2,.75),(.9,.22,.18),(1.2,.3,.65),(1.6,.18,.4)]],
      'relaxation':[experiment(a,t,g) for a,t,g in [(1.5,.4,.15),(1.5,.4,.8),(.95,.5,.2),(.95,.5,.7)]]}
