"""Independent tunneling-amplitude sum over four quasiparticle branches."""
from functools import lru_cache
import numpy as np
from scipy.integrate import quad
from scipy.special import expit

TRUE_PARAMETER = 45.
KB = 1.380649e-23
CHARGE = 1.602176634e-19


def material_gaps(left, right):
    return tuple(1.764*tc*np.tanh(1.74*np.sqrt(tc/t-1)) for t,tc in [(left,1.2),(right,1.9)])


def branch_weight(energy, dl, dr, phase):
    xi_l = np.sqrt(energy**2-dl**2)
    xi_r = np.sqrt(energy**2-dr**2)
    probability = 0.
    for sl in [-1.,1.]:
        ul = np.sqrt((1+sl*xi_l/energy)/2)
        vl = np.sqrt((1-sl*xi_l/energy)/2)
        for sr in [-1.,1.]:
            ur = np.sqrt((1+sr*xi_r/energy)/2)
            vr = np.sqrt((1-sr*xi_r/energy)/2)
            amplitude = ul*ur-np.exp(1j*phase)*vl*vr
            probability += abs(amplitude)**2
    return probability/2


@lru_cache(maxsize=512)
def unit_power(left, right, phase, tolerance=1e-11):
    dl,dr=material_gaps(left,right)
    edge=max(dl,dr)
    # E=edge+z^2 removes the integrable endpoint singularity.
    def integrand(z):
        energy=edge+z*z
        left_jacobian=energy/np.sqrt(energy*energy-dl*dl)
        right_jacobian=energy/np.sqrt(energy*energy-dr*dr)
        probability=branch_weight(energy,dl,dr,phase)
        return 2*z*energy*left_jacobian*right_jacobian*probability*(expit(-energy/left)-expit(-energy/right))
    integral=quad(integrand,0.,np.sqrt(60*max(left,right)),epsabs=tolerance,epsrel=tolerance,limit=160)[0]
    return 2e6*(KB/CHARGE)**2*integral


def predict(experiments, conductance):
    return conductance*np.array([unit_power(e['left_temperature'],e['right_temperature'],e['phase']) for e in experiments])


def experiment(left,right,phase):
    return dict(left_temperature=float(left),right_temperature=float(right),phase=float(phase))


def calibration_inputs():
    return [experiment(a,b,p) for a in np.linspace(.35,1.,8) for b in np.linspace(.3,1.02,9) for p in [-np.pi/2,np.pi/2]]


def hidden_inputs():
    pairs=[(.45,.95),(.75,.3),(.95,.55),(.6,1.),(.3,.8),(.85,.5),(.5,.9),(.8,.4),(.9,.7)]
    return {'aligned_phase':[experiment(a,b,0.) for a,b in pairs],
            'opposite_phase':[experiment(a,b,np.pi) for a,b in pairs],
            'oblique_phase':[experiment(a,b,p) for (a,b),p in zip(pairs,[-.7,.5,.8,-.5,.6,-.8,.7,-.6,.5])]}
