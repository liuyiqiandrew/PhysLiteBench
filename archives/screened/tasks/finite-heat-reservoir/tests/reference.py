"""Energy-space phase volume, independent of the oracle's position quadrature."""
import numpy as np
from scipy.integrate import quad

PARAMETER = 'total_energy'
TRUE_PARAMETER = 1.8


def experiment(k,b,observable='kurtosis'):
    return dict(quadratic=float(k),quartic=float(b),observable=observable)


def calibration_inputs():
    return [experiment(k,0.,'second_moment') for k in np.linspace(.4,2.5,100)]


def hidden_inputs():
    return {'harmonic_shape':[experiment(k,0.) for k in np.linspace(.4,2.5,16)],
            'quartic_shape':[experiment(0.,b) for b in np.linspace(.3,2.,16)],
            'mixed_shape':[experiment(k,b) for k,b in zip(np.linspace(.1,2.,16),np.linspace(2.,.1,16))]}


def predict(experiments,total_energy):
    out = []
    for e in experiments:
        k,b = e['quadratic'],e['quartic']
        # Integrate over probe potential energy u. The remaining five quadratic
        # phase-space coordinates have shell volume proportional to (E-u)^1.5.
        def integrand(u,power):
            q2 = 4*u/(k+np.sqrt(k*k+4*b*u))
            q = np.sqrt(q2)
            jacobian = 1/(k*q+b*q**3)
            return q**power*jacobian*(total_energy-u)**1.5
        moments = [quad(lambda u:integrand(u,n),0,total_energy,epsabs=3e-10,epsrel=3e-10)[0]
                   for n in [0,2,4]]
        second,fourth = moments[1]/moments[0],moments[2]/moments[0]
        out.append(dict(second_moment=second,fourth_moment=fourth,kurtosis=fourth/second**2)[e['observable']])
    return np.array(out)
