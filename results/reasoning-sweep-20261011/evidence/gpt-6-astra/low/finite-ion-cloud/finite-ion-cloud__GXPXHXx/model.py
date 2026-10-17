import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import exprel, hyp2f1, roots_legendre
from functools import lru_cache


@lru_cache(maxsize=None)
def coefficient(gamma,count,ratio):
    if count == 3:
        return three_ion_coefficient(gamma,ratio)
    length = np.log(ratio)
    if count == 1:
        slope = 2-gamma
        return -np.expm1(-2*length)*np.exp(slope*length)/(2*length*exprel(slope*length))
    a,b = 4-3*gamma,2-gamma
    def angular(gap):
        return hyp2f1(-gamma/2,-gamma/2,1,np.exp(-2*gap))
    partition = quad(lambda d: angular(d)*np.exp(b*d)*(length-d)*exprel(a*(length-d)),
                     0,length,epsabs=1e-11,epsrel=1e-11)[0]
    boundary = quad(lambda d: angular(d)*np.exp(b*d+a*(length-d)),
                    0,length,epsabs=1e-11,epsrel=1e-11)[0]
    return -np.expm1(-2*length)*boundary/(4*partition)


def predict_at(experiments,coupling):
    return np.asarray([coefficient(e['temperature_multiplier']*coupling,
                                  e['ion_count'],e['radius_ratio']) for e in experiments])


class Model:
    def __init__(self):
        self.coupling = None

    def fit(self,records):
        """Maximum likelihood fit for independent Gaussian measurements."""
        records = list(records)
        if not records:
            raise ValueError("At least one calibration record is required.")
        experiments = [r['input'] for r in records]
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        if not (np.isfinite(values).all() and np.isfinite(sigma).all()
                and (sigma > 0).all()):
            raise ValueError("Values must be finite and sigma must be positive.")

        def objective(g):
            residual = (predict_at(experiments, g) - values) / sigma
            return np.dot(residual, residual)

        result = minimize_scalar(objective, bounds=(0.7, 1.4),
                                 method='bounded', options={'xatol': 1e-11})
        # A constrained optimum may lie exactly on either endpoint.
        self.coupling = float(min((0.7, result.x, 1.4), key=objective))
        return self

    def predict(self,experiments):
        if self.coupling is None:
            raise ValueError("Fit the model before predicting.")
        return predict_at(experiments,self.coupling)


@lru_cache(maxsize=None)
def nodes(order):
    x,w = roots_legendre(order)
    return (x+1)/2,w/2


def angular_three(gamma, d1, d2, modes=96):
    """Joint angular average, including the closed three-pair correlation.

    For f(t,theta)=|1-t*exp(i*theta)|**gamma, its Fourier
    coefficient is c_k=t**k*(-gamma/2)_k/k! *
    2F1(k-gamma/2,-gamma/2;k+1;t*t). Integrating both independent
    angles leaves sum over k of c_k(t1)c_k(t2)c_k(t1*t2).
    """
    p = gamma / 2
    gaps = np.asarray([d1, d2, d1+d2])
    z = np.exp(-2*gaps)
    angular = np.prod(hyp2f1(-p, -p, 1, z), axis=0)
    prefactor = 1.0
    for k in range(1, modes+1):
        prefactor *= (k-1-p)/k
        if prefactor == 0:  # Even integer gamma gives a finite polynomial.
            break
        term = prefactor**3 * np.exp(-2*k*(d1+d2))
        term *= np.prod(hyp2f1(k-p, -p, k+1, z), axis=0)
        angular += 2*term
    return angular


def three_ion_coefficient(gamma,ratio,order=40, modes=96):
    """Outer-wall contact pressure divided by N kT / accessible area.

    Order log radii as t, t+d1, t+d1+d2. The common radial
    translation integrates analytically from 0 to L-d1-d2.
    Differentiating this integral with respect to L gives the boundary
    integral. Thus phi=(1-exp(-2L))*d(log Z)/dL/(2N).
    All dimensional scale factors cancel, including disk_radius.
    """
    length = np.log(ratio)
    x,w = nodes(order)
    d1 = np.broadcast_to((length*x)[:,None],(order,order)).ravel()
    d2 = ((length-length*x)[:,None]*x[None,:]).ravel()
    weights = (length*(length-length*x)[:,None]*w[:,None]*w[None,:]).ravel()
    angular = angular_three(gamma, d1, d2, modes)
    span = length-d1-d2
    slope = 6-6*gamma
    weight = weights*angular*np.exp((4-3*gamma)*d1+(2-gamma)*d2)
    partition = np.dot(weight,span*exprel(slope*span))
    boundary = np.dot(weight,np.exp(slope*span))
    return -np.expm1(-2*length)*boundary/(6*partition)
