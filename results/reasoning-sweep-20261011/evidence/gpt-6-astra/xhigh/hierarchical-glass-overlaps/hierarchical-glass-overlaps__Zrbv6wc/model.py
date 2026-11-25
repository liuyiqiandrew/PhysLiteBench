"""Equilibrium calibration and configuration coincidences."""
import math
from numbers import Real, Integral
import numpy as np

WIDTH_INTERVAL = (0.8, 1.2)
ENTROPIES = (0.3, 0.7)
VARIANCES = (0.7, 0.3)


def level_parameter(beta, width, level):
    critical = math.sqrt(2*ENTROPIES[level]/VARIANCES[level])/width
    return min(1.0, critical/beta)


def participation(m, replicas):
    result = 1.0
    for j in range(1, replicas):
        result *= (j-m)/j
    return result


def leaf_partition(q, alpha, blocks):
    """Probability of one labelled leaf partition within a common family.

    The two freezing thresholds are ordered, so the limiting Gibbs weights
    form a two-level Poisson cascade. Family masses have law PD(q, 0).
    Their internal weights have the tilted law PD(alpha, -q), rather than
    PD(alpha, 0): a family's internal partition sum also affects its mass.
    Multiplying that partition law by the family participation cancels
    the (1-q) rising factorial, giving the product below.

    The formula extends to alpha=1 (diffuse leaves) and q=alpha=1
    (diffuse families). Blocks are fixed replica sets, so no combinatorial
    multiplicity is included.
    """
    n = sum(blocks)
    k = len(blocks)
    result = 1.0/math.factorial(n-1)
    for j in range(1, k):
        result *= j*alpha-q
    for size in blocks:
        for j in range(1,size):
            result *= j-alpha
    return result


def coincidence(beta, replicas, observable, width, blocks=None):
    q = level_parameter(beta,width,0)
    alpha = level_parameter(beta,width,1)
    if observable == 'parent_collision':
        return participation(q,replicas)
    if observable == 'state_collision':
        return participation(alpha,replicas)
    return leaf_partition(q,alpha,blocks)


def calibration_beta(row):
    beta = float(row['input']['beta'])
    if not math.isfinite(beta) or not .2 <= beta <= .6:
        raise ValueError('Outside the stated calibration inverse-temperature range.')
    return beta


class Model:
    def __init__(self):
        self.width = None

    def _parameter(self, width):
        if width is not None and (isinstance(width,bool) or not isinstance(width,Real)):
            raise ValueError('Supply a real material scale.')
        value = self.width if width is None else float(width)
        if value is None or not math.isfinite(value) or not .8 <= value <= 1.2:
            raise ValueError('Fit or supply width in [0.8,1.2].')
        return value

    def fit(self, records):
        """Fit width by constrained Gaussian maximum likelihood.

        All calibration temperatures are above both freezing temperatures,
        even at width=1.2. Thus the energy per size is -beta*width**2,
        and weighted least squares is linear in the squared width.
        """
        readings = []
        for row in records:
            beta = calibration_beta(row)
            measured = float(row['value'])
            sigma = float(row['sigma'])
            if not math.isfinite(measured) or not math.isfinite(sigma) or sigma <= 0:
                raise ValueError('Calibration values must be finite and sigma positive and finite.')
            readings.append((beta, measured, sigma))
        if not readings:
            raise ValueError('Supply at least one calibration record.')

        # A common rescaling of inverse variances leaves the fit unchanged
        # and avoids overflow when measurement uncertainties are very small.
        scale = min(sigma for _, _, sigma in readings)
        weights = [(scale/sigma)**2 for _, _, sigma in readings]
        numerator = math.fsum(-weight*beta*measured
                              for (beta, measured, _), weight in zip(readings, weights))
        denominator = math.fsum(weight*beta*beta
                                for (beta, _, _), weight in zip(readings, weights))
        squared_width = numerator/denominator
        lower, upper = WIDTH_INTERVAL
        self.width = math.sqrt(max(lower*lower, min(upper*upper, squared_width)))
        return self

    def predict_calibration(self, records, width=None):
        value = self._parameter(width)
        return np.asarray([-calibration_beta(row)*value*value for row in records],dtype=float)

    def predict(self, experiments, width=None):
        value = self._parameter(width)
        result = []
        for row in experiments:
            beta = float(row['beta'])
            replicas = row['replicas']
            observable = row['observable']
            if not math.isfinite(beta) or not .2 <= beta <= 6:
                raise ValueError('Outside the stated inverse-temperature range.')
            if isinstance(replicas,bool) or not isinstance(replicas,Integral) or not 2 <= replicas <= 6:
                raise ValueError('Use an integer number of replicas from 2 through 6.')
            if observable not in ('parent_collision','state_collision','leaf_partition'):
                raise ValueError('Unknown observable.')
            blocks = None
            if observable == 'leaf_partition':
                blocks = list(row['partition'])
                if (not blocks or any(isinstance(n,bool) or not isinstance(n,Integral) or n<1 for n in blocks)
                        or sum(blocks)!=replicas):
                    raise ValueError('Partition sizes must be positive integers summing to replicas.')
            result.append(coincidence(beta,replicas,observable,value,blocks))
        return np.asarray(result,dtype=float)
