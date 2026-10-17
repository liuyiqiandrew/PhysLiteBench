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
                                  e['ion_count'],e['radius_ratio'])
                       for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.coupling = None

    def fit(self,records):
        if not records:
            raise ValueError('At least one calibration record is required.')

        experiments = [record['input'] for record in records]
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)
        if (not np.all(np.isfinite(values)) or
                not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0)):
            raise ValueError('Calibration values and sigmas must be finite, with positive sigmas.')

        def objective(g):
            residual = (predict_at(experiments, float(g)) - values) / sigmas
            return float(np.dot(residual, residual))

        result = minimize_scalar(objective, bounds=(0.7, 1.4), method='bounded',
                                 options={'xatol': 1e-11})
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError('Unable to fit the reference electrostatic coupling.')
        self.coupling = float(result.x)
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.coupling)


@lru_cache(maxsize=None)
def nodes(order):
    x,w = roots_legendre(order)
    return (x+1)/2,w/2


def three_ion_coefficient(gamma,ratio,order=32):
    """Return the finite-cell coefficient for three ions.

    After sorting the three logarithmic radii, d1 and d2 are the adjacent
    radial gaps.  The three relative angles are not independent: if alpha
    and beta are the adjacent angular gaps, the third gap is alpha+beta.
    The angular integral is therefore a coupled two-angle integral.  The
    Fourier coefficients below evaluate that integral as

        <h1(alpha) h2(beta) h3(alpha+beta)>
          = c1[0] c2[0] c3[0] + 2 sum_n c1[n] c2[n] c3[n].

    Here h(d,theta) is the angular part of a pair distance raised to gamma.
    """
    length = np.log(ratio)
    x,w = nodes(order)
    d1 = np.broadcast_to((length*x)[:,None],(order,order)).ravel()
    d2 = ((length-length*x)[:,None]*x[None,:]).ravel()
    weights = (length*(length-length*x)[:,None]*w[:,None]*w[None,:]).ravel()

    # For q=exp(-d),
    # h(d,theta)=(1-q exp(i theta))^(gamma/2)
    #             (1-q exp(-i theta))^(gamma/2).
    # Its non-negative Fourier coefficients are
    # c_n = q^n (-p)_n/n! * 2F1(n-p,-p;n+1;q^2), p=gamma/2.
    # 128 modes are ample even at the smallest allowed gamma; the product
    # of three coefficients makes the omitted tail much smaller than the
    # radial quadrature error.
    modes = 128
    p = gamma/2
    n = np.arange(modes + 1)
    fourier = []
    for gap in (d1, d2, d1+d2):
        q = np.exp(-gap)
        coeff = np.ones((len(gap), modes + 1))
        for mode in range(modes):
            coeff[:, mode+1] = (coeff[:, mode] * q *
                                (mode-p)/(mode+1))
        coeff *= hyp2f1(n[None, :] - p, -p, n[None, :] + 1,
                        q[:, None]**2)
        fourier.append(coeff)

    angular = (fourier[0][:, 0] * fourier[1][:, 0] * fourier[2][:, 0] +
               2*np.sum(fourier[0][:, 1:] * fourier[1][:, 1:] *
                        fourier[2][:, 1:], axis=1))
    span = length-d1-d2
    slope = 6-6*gamma
    weight = weights*angular*np.exp((4-3*gamma)*d1+(2-gamma)*d2)
    partition = np.dot(weight,span*exprel(slope*span))
    boundary = np.dot(weight,np.exp(slope*span))
    return -np.expm1(-2*length)*boundary/(6*partition)
