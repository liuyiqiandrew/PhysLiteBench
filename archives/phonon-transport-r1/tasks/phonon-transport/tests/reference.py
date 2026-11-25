"""Direct angular-ordinate transport with local energy/momentum projections."""
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss,legval
from scipy.linalg import expm

PARAMETER='resistive_rate'
TRUE_PARAMETER=.3


@lru_cache(8)
def angles(order):
    mu,w=leggauss(order)
    w=w/2
    isotropic=np.outer(np.ones(order),w)
    momentum=3*np.outer(mu,w*mu)
    return mu,w,isotropic,momentum


def predict(experiments,resistive_rate=TRUE_PARAMETER,order=80):
    mu,w,isotropic,momentum=angles(order);out=[]
    for e in experiments:
        initial=legval(mu,e['initial']).astype(complex)
        resistive=resistive_rate*(isotropic-np.eye(order))
        normal=e['normal_rate']*(isotropic+momentum-np.eye(order))
        generator=-1j*e['wavenumber']*np.diag(mu)+resistive+normal
        final=expm(generator*e['time'])@initial
        polynomial=legval(mu,[0]*e['moment']+[1])
        value=(w*polynomial)@final
        out.append(float(value.real if e['quadrature']=='cosine' else -value.imag))
    return np.array(out)


def reading(time,wavenumber=0.,normal_rate=.4,initial=(0.,0.,.12),moment=2,quadrature='cosine'):
    return dict(time=float(time),wavenumber=float(wavenumber),normal_rate=float(normal_rate),
                initial=list(initial),moment=int(moment),quadrature=quadrature)


def calibration_inputs():
    return [reading(t,normal_rate=normal) for normal in [.2,.5,1.] for t in np.geomspace(.05,3.,48)]


def hidden_inputs():
    return {'energy_wave':[reading(t,1.2,2.5,(.12,0.,0.),0) for t in np.linspace(.4,6.,24)],
            'heat_current':[reading(t,1.5,2.,(.12,0.,0.),1,'sine') for t in np.linspace(.2,5.,24)],
            'anisotropy_transfer':[reading(t,1.3,2.5,(0.,0.,.12),0) for t in np.linspace(.4,6.,24)]}
