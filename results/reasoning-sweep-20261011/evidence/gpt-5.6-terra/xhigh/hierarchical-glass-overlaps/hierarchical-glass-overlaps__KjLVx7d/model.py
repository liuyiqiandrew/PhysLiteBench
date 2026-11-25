"""Equilibrium calibration and coincidence probabilities for the GREM.

The limiting Gibbs weights form a two-level Ruelle probability cascade.  In
particular, the partition of replicas into leaf labels is Poisson--Dirichlet
with the second-level parameter, and its leaf blocks are coalesced into
families with the ratio of the two cascade parameters.
"""
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
    """Return a disorder-averaged replica event in the infinite-size limit."""
    q = level_parameter(beta,width,0)
    alpha = level_parameter(beta,width,1)
    parent = participation(q,replicas)
    if observable == 'parent_collision':
        return parent
    if observable == 'state_collision':
        # A leaf is already a terminal atom of the cascade; multiplying by
        # the parent participation would double-count the first split.
        return participation(alpha,replicas)

    # The leaf partition has PD(alpha) EPPF.  Given its k distinct leaf
    # blocks, their family labels have PD(q / alpha) EPPF, so the requested
    # event asks that those k blocks coalesce into one family.
    return leaf_partition(alpha,blocks)*participation(q/alpha,len(blocks))


def _real(value, message):
    """Convert a finite, non-Boolean real to float, or raise ValueError."""
    if isinstance(value,bool) or not isinstance(value,Real):
        raise ValueError(message)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(message)
    return value


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
        """Fit width by inverse-variance weighted least squares.

        Across the specified calibration range both hierarchy levels are in
        their high-temperature phase.  The limiting energy density is hence
        ``-beta * width**2``.  Fitting its squared width makes the likelihood
        a one-parameter weighted linear regression, followed by projection
        onto the stated material interval.
        """
        numerator = 0.0
        denominator = 0.0
        sigma_scale = None
        count = 0
        for row in records:
            try:
                beta = _real(row['input']['beta'], 'Calibration beta must be a finite real number.')
                measured = _real(row['value'], 'Calibration value must be a finite real number.')
                sigma = _real(row['sigma'], 'Calibration sigma must be a finite positive real number.')
            except (KeyError, TypeError):
                raise ValueError('Each calibration record needs input.beta, value, and sigma.') from None
            if not .2 <= beta <= .6:
                raise ValueError('Outside the stated calibration inverse-temperature range.')
            if sigma <= 0.0:
                raise ValueError('Calibration sigma must be a finite positive real number.')
            # Only relative inverse variances affect the estimate.  Keeping
            # them scaled to the smallest sigma seen avoids overflow for
            # otherwise valid, very small floating-point uncertainties.
            if sigma_scale is None or sigma < sigma_scale:
                if sigma_scale is not None:
                    rescale = (sigma/sigma_scale)**2
                    numerator *= rescale
                    denominator *= rescale
                sigma_scale = sigma
            relative_weight = (sigma_scale/sigma)**2
            numerator -= beta*measured*relative_weight
            denominator += beta*beta*relative_weight
            count += 1
        if not count:
            raise ValueError('At least one calibration record is required.')

        squared_width = numerator/denominator
        squared_width = min(WIDTH_INTERVAL[1]**2,
                            max(WIDTH_INTERVAL[0]**2,squared_width))
        self.width = math.sqrt(squared_width)
        return self

    def predict_calibration(self, records, width=None):
        value = self._parameter(width)
        result = []
        for row in records:
            try:
                beta = _real(row['input']['beta'], 'Calibration beta must be a finite real number.')
            except (KeyError, TypeError):
                raise ValueError('Each calibration record needs input.beta.') from None
            if not .2 <= beta <= .6:
                raise ValueError('Outside the stated calibration inverse-temperature range.')
            result.append(-beta*value*value)
        return np.asarray(result,dtype=float)

    def predict(self, experiments, width=None):
        value = self._parameter(width)
        result = []
        for row in experiments:
            try:
                beta = _real(row['beta'], 'Outside the stated inverse-temperature range.')
                replicas = row['replicas']
                observable = row['observable']
            except (KeyError, TypeError):
                raise ValueError('Each experiment needs beta, replicas, and observable.') from None
            if not .2 <= beta <= 6:
                raise ValueError('Outside the stated inverse-temperature range.')
            if isinstance(replicas,bool) or not isinstance(replicas,Integral) or not 2 <= replicas <= 6:
                raise ValueError('Use an integer number of replicas from 2 through 6.')
            if observable not in ('parent_collision','state_collision','leaf_partition'):
                raise ValueError('Unknown observable.')
            blocks = None
            if observable == 'leaf_partition':
                try:
                    blocks = list(row['partition'])
                except (KeyError, TypeError):
                    raise ValueError('Partition sizes must be positive integers summing to replicas.') from None
                if (not blocks or any(isinstance(n,bool) or not isinstance(n,Integral) or n<1 for n in blocks)
                        or sum(blocks)!=replicas):
                    raise ValueError('Partition sizes must be positive integers summing to replicas.')
            result.append(coincidence(beta,replicas,observable,value,blocks))
        return np.asarray(result,dtype=float)
