"""Fourier transform of the vacuum correlator along the specified worldlines."""
import numpy as np
from scipy.integrate import quad

TRUE_PARAMETER = .00011


def spectrum(omega, acceleration, separation, tolerance=1e-11, shift=1.):
    a=acceleration; L=separation
    eta=shift*(min(1/abs(omega),np.pi/a) if a else 1/abs(omega))
    def wightman(u):
        z=u-1j*eta
        if a and a*u>700:return 0j
        interval=4*np.sinh(a*z/2)**2/a**2 if a else z*z
        return -1/(4*np.pi**2*(interval-L*L))
    c,ce=quad(lambda u:wightman(u).real,0,np.inf,weight='cos',wvar=omega,epsabs=tolerance,limlst=160,limit=250)
    s,se=quad(lambda u:wightman(u).imag,0,np.inf,weight='sin',wvar=omega,epsabs=tolerance,limlst=160,limit=250)
    return 2*np.exp(omega*eta)*(c-s)


def predict(experiments, coupling_squared, tolerance=1e-11, shift=1.):
    values = []
    for e in experiments:
        local = spectrum(e['gap'], e['acceleration'], 0., tolerance, shift)
        if e['preparation'] == 'single':
            values.append(local)
        else:
            cross = spectrum(e['gap'], e['acceleration'], e['separation'], tolerance, shift)
            sign = 1 if e['preparation'] == 'symmetric' else -1
            values.append(local+sign*cross)
    return coupling_squared*np.asarray(values)


def experiment(**changes):
    row = dict(preparation='symmetric', gap=.8, acceleration=6., separation=1.)
    row.update(changes)
    return row


def calibration_inputs():
    return ([experiment(preparation='single', gap=w, acceleration=a)
             for w in [.5, .9, 1.4, 2.] for a in [0., 2., 5., 10.]]
            +[experiment(preparation=p, gap=w, acceleration=0., separation=d)
              for p in ['symmetric', 'antisymmetric'] for w in [.7, 1.5] for d in [.4, 1.2, 2.5]])


def hidden_inputs():
    return {
        'shared_anchors': [experiment(preparation='single', gap=w, acceleration=a)
                           for w in [.6, 1.6] for a in [1., 4., 8.]]
                          +[experiment(preparation=p, gap=1.2, acceleration=0., separation=d)
                            for p in ['symmetric', 'antisymmetric'] for d in [.8, 1.8]],
        'symmetric_pairs': [experiment(gap=w, acceleration=a, separation=d)
                            for w in [.5, .9] for a in [6., 10.] for d in [.7, 1.2, 1.8]],
        'antisymmetric_pairs': [experiment(preparation='antisymmetric', gap=w, acceleration=a, separation=d)
                                for w in [.5, .9] for a in [6., 10.] for d in [.7, 1.2, 1.8]],
        'frequency_response': [experiment(gap=w, acceleration=a, separation=1.1)
                               for w in [.5, 1., 1.5, 2.] for a in [4., 7.]],
    }
