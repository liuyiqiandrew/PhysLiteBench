import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import exprel, hyp2f1, roots_legendre
from functools import lru_cache


@lru_cache(maxsize=4096)
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
        records = list(records)
        if not records:
            raise ValueError("At least one calibration record is required.")
        experiments = [r["input"] for r in records]
        values = np.asarray([r["value"] for r in records], dtype=float)
        sigma = np.asarray([r["sigma"] for r in records], dtype=float)
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(sigma) & (sigma > 0)):
            raise ValueError("Calibration values must be finite and sigma positive.")

        def loss(coupling):
            residual = (predict_at(experiments, coupling) - values) / sigma
            return np.dot(residual, residual)

        result = minimize_scalar(loss, bounds=(0.7, 1.4), method="bounded",
                                 options={"xatol": 1e-10})
        # Include endpoints: a constrained maximum likelihood fit can lie there.
        self.coupling = float(min((0.7, result.x, 1.4), key=loss))
        return self

    def predict(self,experiments):
        if self.coupling is None:
            raise ValueError("Fit the model before predicting.")
        return predict_at(experiments,self.coupling)


@lru_cache(maxsize=None)
def nodes(order):
    x,w = roots_legendre(order)
    return (x+1)/2,w/2


def three_ion_coefficient(gamma, ratio, order=64, angular_order=512):
    """Canonical wall contact density, integrating ordered log radii.

    Put log(r/a) = (s, s+d1, s+d1+d2). Integrating s analytically
    gives span*exprel((6-6*gamma)*span); fixing the largest radius
    at the wall instead gives exp((6-6*gamma)*span).

    For pair angular factors f_t(theta)=|1-t*exp(i*theta)|**gamma,
    the joint angular average is sum_k c_k(t1)c_k(t2)c_k(t1*t2).
    This retains the triangular angular correlation. Fourier coefficients
    are evaluated by periodic quadrature, in batches to bound memory use.
    """
    length = np.log(ratio)
    x,w = nodes(order)
    d1 = np.broadcast_to((length*x)[:,None],(order,order)).ravel()
    d2 = ((length-length*x)[:,None]*x[None,:]).ravel()
    weights = (length*(length-length*x)[:,None]*w[:,None]*w[None,:]).ravel()
    angular = np.empty(len(d1))
    sin2 = np.sin(np.pi*np.arange(angular_order)/angular_order)**2
    for start in range(0, len(d1), 256):
        stop = min(start+256, len(d1))
        product = np.ones((stop-start, angular_order//2+1))
        for gap in (d1[start:stop], d2[start:stop],
                    d1[start:stop]+d2[start:stop]):
            t = np.exp(-gap[:,None])
            distance2 = (-np.expm1(-gap[:,None]))**2 + 4*t*sin2
            fourier = np.fft.rfft(distance2**(gamma/2), axis=1).real/angular_order
            product *= fourier
        angular[start:stop] = (product[:,0] + 2*product[:,1:-1].sum(axis=1)
                               + product[:,-1])
    span = length-d1-d2
    slope = 6-6*gamma
    weight = weights*angular*np.exp((4-3*gamma)*d1+(2-gamma)*d2)
    partition = np.dot(weight,span*exprel(slope*span))
    boundary = np.dot(weight,np.exp(slope*span))
    # phi = (1-exp(-2L))/(2N) * d(log Z)/dL.
    # The absolute disk radius cancels from this dimensionless ratio.
    return -np.expm1(-2*length)*boundary/(6*partition)
