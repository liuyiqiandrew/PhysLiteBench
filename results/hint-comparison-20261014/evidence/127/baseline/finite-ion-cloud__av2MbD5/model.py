import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import exprel, hyp2f1, roots_legendre
from functools import lru_cache


def _angular_average(gamma, gap):
    """Angular average of a pair-distance factor at logarithmic gap gap."""
    return hyp2f1(-gamma / 2, -gamma / 2, 1, np.exp(-2 * gap))


@lru_cache(maxsize=None)
def coefficient(gamma, count, ratio):
    """Finite-cell osmotic coefficient for a specified dimensionless coupling.

    After ordering the radial logarithmic coordinates, all absolute angles can
    be integrated analytically.  Differentiating the resulting canonical
    integral at the outer boundary gives the ``boundary / partition`` forms
    below.  Factors independent of the outer radius (including the disk
    radius and the ordering multiplicity) cancel.
    """
    length = np.log(ratio)
    area_factor = -np.expm1(-2 * length)  # 1 - (a / R)**2

    if count == 1:
        slope = 2 - gamma
        partition = length * exprel(slope * length)
        boundary = np.exp(slope * length)
        return area_factor * boundary / (2 * partition)

    if count == 2:
        # d is the gap between the outer and inner ion.  The remaining
        # coordinate has exponent 4 - 3*gamma.
        inner_slope, gap_slope = 4 - 3 * gamma, 2 - gamma

        def integrand_partition(gap):
            remaining = length - gap
            return (_angular_average(gamma, gap) * np.exp(gap_slope * gap)
                    * remaining * exprel(inner_slope * remaining))

        def integrand_boundary(gap):
            remaining = length - gap
            return (_angular_average(gamma, gap) * np.exp(
                gap_slope * gap + inner_slope * remaining))

        partition = quad(integrand_partition, 0, length,
                         epsabs=1e-11, epsrel=1e-11)[0]
        boundary = quad(integrand_boundary, 0, length,
                        epsabs=1e-11, epsrel=1e-11)[0]
        return area_factor * boundary / (4 * partition)

    if count == 3:
        return three_ion_coefficient(gamma, ratio)

    raise ValueError("ion_count must be 1, 2, or 3")


def predict_at(experiments,coupling):
    return np.asarray([coefficient(e['temperature_multiplier']*coupling,
                                  e['ion_count'],e['radius_ratio']) for e in experiments])


class Model:
    def __init__(self):
        self.coupling = None

    def fit(self,records):
        """Fit the shared reference coupling by Gaussian weighted least squares."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record['input'] for record in records]
        observed = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)
        if (not np.isfinite(observed).all() or not np.isfinite(sigma).all()
                or np.any(sigma <= 0)):
            raise ValueError("calibration values must be finite and sigma must be positive")

        def objective(g):
            residual = (predict_at(experiments, g) - observed) / sigma
            return float(np.dot(residual, residual))

        result = minimize_scalar(objective, bounds=(0.7, 1.4), method='bounded',
                                 options={'xatol': 1e-11})
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError("could not fit electrostatic coupling")
        self.coupling = float(result.x)
        return self

    def predict(self,experiments):
        if self.coupling is None:
            raise RuntimeError("fit must be called before predict")
        return predict_at(experiments,self.coupling)


@lru_cache(maxsize=None)
def nodes(order):
    x,w = roots_legendre(order)
    return (x+1)/2,w/2


def three_ion_coefficient(gamma,ratio,order=48):
    length = np.log(ratio)
    x,w = nodes(order)
    d1 = np.broadcast_to((length*x)[:,None],(order,order)).ravel()
    d2 = ((length-length*x)[:,None]*x[None,:]).ravel()
    weights = (length*(length-length*x)[:,None]*w[:,None]*w[None,:]).ravel()
    angular = np.ones(len(d1))
    for gap in [d1,d2,d1+d2]:
        angular *= _angular_average(gamma, gap)
    span = length-d1-d2
    slope = 6-6*gamma
    weight = weights*angular*np.exp((4-3*gamma)*d1+(2-gamma)*d2)
    partition = np.dot(weight,span*exprel(slope*span))
    boundary = np.dot(weight,np.exp(slope*span))
    return -np.expm1(-2*length)*boundary/(6*partition)
