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


def leaf_partition(alpha, blocks):
    n = sum(blocks)
    k = len(blocks)
    result = alpha**(k-1)*math.factorial(k-1)/math.factorial(n-1)
    for size in blocks:
        for j in range(1,size):
            result *= j-alpha
    return result


def coincidence(beta, replicas, observable, width, blocks=None):
    """Return the disorder averaged replica event in the N -> infinity limit.

    The two frozen levels give independent Poisson--Dirichlet factors.  The
    family partition sum does bias which family is selected, but after that
    (fractional-power) bias the stable descendant partition sum is independent
    of its normalized descendant weights.  Consequently the bias changes no
    further factor here; it is important that the descendant factor is still
    included for every labelled block partition.
    """
    q = level_parameter(beta,width,0)
    alpha = level_parameter(beta,width,1)
    parent = participation(q,replicas)
    if observable == 'parent_collision':
        return parent
    if observable == 'state_collision':
        return parent*participation(alpha,replicas)
    return parent*leaf_partition(alpha,blocks)


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
        """Fit width by weighted least squares in the high-temperature branch.

        All calibration temperatures are below the first transition for every
        allowed width.  Thus the limiting equilibrium energy density is
        ``-beta * width**2``.  Fitting width squared makes this a linear,
        one-parameter weighted regression and avoids a needless nonlinear
        optimizer.
        """
        rows = list(records)
        if not rows:
            raise ValueError('Calibration records are required to fit width.')

        numerator = 0.0
        denominator = 0.0
        for row in rows:
            try:
                beta = row['input']['beta']
                measured = row['value']
                sigma = row['sigma']
            except (KeyError, TypeError):
                raise ValueError('Malformed calibration record.') from None

            if (isinstance(beta, bool) or not isinstance(beta, Real)
                    or not math.isfinite(float(beta)) or not .2 <= float(beta) <= .6):
                raise ValueError('Calibration beta is outside the stated range.')
            if (isinstance(measured, bool) or not isinstance(measured, Real)
                    or not math.isfinite(float(measured))):
                raise ValueError('Calibration value must be finite and real.')
            if (isinstance(sigma, bool) or not isinstance(sigma, Real)
                    or not math.isfinite(float(sigma)) or float(sigma) <= 0):
                raise ValueError('Calibration sigma must be positive and finite.')

            beta = float(beta)
            measured = float(measured)
            weight = 1.0 / float(sigma)**2
            numerator -= weight * beta * measured
            denominator += weight * beta * beta

        theta = numerator / denominator
        theta = min(WIDTH_INTERVAL[1]**2, max(WIDTH_INTERVAL[0]**2, theta))
        self.width = math.sqrt(theta)
        return self

    def predict_calibration(self, records, width=None):
        value = self._parameter(width)
        return np.asarray([-float(row['input']['beta'])*value*value for row in records],dtype=float)

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
