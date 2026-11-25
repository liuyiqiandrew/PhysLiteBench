"""Full two-dimensional moving-basis Rayleigh-Ritz reference."""
import numpy as np
from scipy.linalg import eigh
from scipy.special import roots_legendre

TRUE_PARAMETER = 1.07


def finite(e,epsilon,modes=40,transverse=7,points=220):
    q,w=roots_legendre(points);x=(q+1)/2;w=w/2
    qt,wt=roots_legendre(100);u=qt/2;wt=wt/2
    n=np.arange(1,modes+1)
    phi=np.sqrt(2)*np.sin(np.pi*x[:,None]*n)
    dphi=np.sqrt(2)*np.pi*n*np.cos(np.pi*x[:,None]*n)
    t=np.arange(1,transverse+1)
    chi=np.sqrt(2)*np.sin(np.pi*(u[:,None]+.5)*t)
    dchi=np.sqrt(2)*np.pi*t*np.cos(np.pi*(u[:,None]+.5)*t)
    dilation=u[:,None]*dchi+.5*chi
    B=chi.T@(wt[:,None]*dilation)
    G=dilation.T@(wt[:,None]*dilation)
    a=np.exp(e['amplitude']*np.cos(2*np.pi*x)+e['harmonic']*np.cos(4*np.pi*x+e['phase']))
    r=-2*np.pi*e['amplitude']*np.sin(2*np.pi*x)-4*np.pi*e['harmonic']*np.sin(4*np.pi*x+e['phase'])
    grad=phi.T@((w*r)[:,None]*dphi)
    h=np.kron(np.diag((np.pi*n)**2),np.eye(transverse))-np.kron(grad.T,B)-np.kron(grad,B.T)
    h+=np.kron(phi.T@((w*r*r)[:,None]*phi),G)
    h+=np.kron(phi.T@((w/(epsilon*a)**2)[:,None]*phi),np.diag(np.pi**2*(t*t-1)))
    h+=np.kron(phi.T@((w*e['well']*np.cos(2*np.pi*x))[:,None]*phi),np.eye(transverse))
    v=eigh(h,eigvals_only=True,subset_by_index=(0,e['level']))
    return v[e['level']]-v[0]

def predict(experiments, gain, epsilon=.03, modes=48, transverse=7):
    widths = epsilon/np.array([1., 2., 4.])
    values = []
    for e in experiments:
        gaps = [finite(e, ep, modes=modes, transverse=transverse) for ep in widths]
        values.append(np.polynomial.polynomial.polyfit(widths**2, gaps, 2)[0])
    return gain*np.asarray(values)


def experiment(**changes):
    row = dict(amplitude=.6, harmonic=.25, phase=.5, well=5., level=1)
    row.update(changes)
    return row


def calibration_inputs():
    return [experiment(amplitude=0., harmonic=0., well=w, level=n)
            for w in [-18., -6., 6., 18.] for n in [1, 2, 3]]


def hidden_inputs():
    return {
        'constant_width': [experiment(amplitude=0., harmonic=0., well=w, level=n)
                           for w in [-12., 0., 12.] for n in [1, 2, 3]],
        'fundamental_shape': [experiment(amplitude=a, harmonic=0., well=w, level=1)
                              for a in [.55, .7, .8] for w in [-10., 10.]],
        'mixed_shape': [experiment(amplitude=.6, harmonic=.3, phase=ph, well=5., level=n)
                        for ph in [0., .7, 1.2] for n in [1, 2]],
        'shifted_shape': [experiment(amplitude=.55, harmonic=.35, phase=ph, well=-5., level=2)
                          for ph in [2.2, 2.6, 3.0]],
        'higher_levels': [experiment(amplitude=.7, harmonic=.4, phase=ph, well=w, level=3)
                          for ph in [0., .5] for w in [-10., 10.]],
    }
