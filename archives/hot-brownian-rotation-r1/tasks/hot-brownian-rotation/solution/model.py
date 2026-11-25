from functools import lru_cache
import numpy as np
from scipy.integrate import quad


def impedance(e,viscosity):
    a=e['radius']
    z=a*np.sqrt(-1j*e['omega']/viscosity)
    return 8*np.pi*viscosity*a**3*(1+z*z/(3*(1+z)))


@lru_cache(128)
def thermal_weight(frequency):
    if frequency==0:return .75
    k=np.sqrt(-1j*frequency)
    def dissipation(r):
        flow=np.exp(-k*(r-1))*(1+k*r)/((1+k)*r*r)
        strain=flow*(-3/r-k+k/(1+k*r))
        return r*r*abs(strain)**2
    denominator=quad(dissipation,1,np.inf,epsabs=1e-10,epsrel=2e-11)[0]
    numerator=quad(lambda r:dissipation(r)/r,1,np.inf,epsabs=1e-10,epsrel=2e-11)[0]
    return numerator/denominator


def spectrum(e,viscosity):
    temperature=e['ambient']+thermal_weight(e['omega']*e['radius']**2/viscosity)*e['rise']
    return float(2*temperature*impedance(e,viscosity).real)


def predict_at(experiments,viscosity):
    return np.asarray([spectrum(e,viscosity) for e in experiments])


class Model:
    def __init__(self):self.viscosity=None

    def fit(self,records):
        records=list(records)
        basis=predict_at([r['input'] for r in records],1.)
        value=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        self.viscosity=float(np.dot(basis/sigma,value/sigma)/np.dot(basis/sigma,basis/sigma))
        return self

    def predict(self,experiments):return predict_at(experiments,self.viscosity)
