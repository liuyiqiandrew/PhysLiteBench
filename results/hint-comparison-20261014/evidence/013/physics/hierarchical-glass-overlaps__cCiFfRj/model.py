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


def selected_leaf_partition(parent_parameter, leaf_parameter, blocks):
    """Probability of a specified leaf partition within one selected family.

    Selecting a family from the normalized cascade tilts its descendant
    partition function by ``parent_parameter``.  Consequently its normalized
    descendant weights have the PD(leaf_parameter, -parent_parameter) EPPF,
    rather than an independent PD(leaf_parameter) law.
    """
    n = sum(blocks)
    result = 1.0
    for j in range(1, n):
        result /= j - parent_parameter
    for j in range(1, len(blocks)):
        result *= j * leaf_parameter - parent_parameter
    for size in blocks:
        for j in range(1, size):
            result *= j - leaf_parameter
    return result


def joint_leaf_partition(parent_parameter, leaf_parameter, blocks):
    """Specified leaf partition, including the event of one common parent."""
    n = sum(blocks)
    result = 1.0 / math.factorial(n - 1)
    for j in range(1, len(blocks)):
        result *= j * leaf_parameter - parent_parameter
    for size in blocks:
        for j in range(1, size):
            result *= j - leaf_parameter
    return result


def coincidence(beta, replicas, observable, width, blocks=None):
    q = level_parameter(beta,width,0)
    alpha = level_parameter(beta,width,1)
    parent = participation(q,replicas)
    if observable == 'parent_collision':
        return parent
    if observable == 'state_collision':
        # The state weights are themselves PD(alpha); multiplying two
        # independent participation ratios would incorrectly condition on a
        # typical, rather than Gibbs-selected, family.
        return participation(alpha,replicas)
    # This is parent * PD(alpha, -q), algebraically simplified so that the
    # unfrozen boundary q=1 remains finite (the conditional law there is not
    # defined, while the joint event is).
    return joint_leaf_partition(q,alpha,blocks)


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
        """Maximum-likelihood estimate under the supplied Gaussian errors."""
        numerator = 0.0
        denominator = 0.0
        count = 0
        for row in records:
            try:
                beta = float(row['input']['beta'])
                observed = float(row['value'])
                sigma = float(row['sigma'])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('Calibration records need finite beta, value, and sigma.') from exc
            if (not math.isfinite(beta) or not .2 <= beta <= .6
                    or not math.isfinite(observed) or not math.isfinite(sigma) or sigma <= 0):
                raise ValueError('Calibration records are outside their stated range.')
            inverse_variance = 1.0 / (sigma * sigma)
            # The calibration curve is -beta * width**2.  It is linear in
            # t=width**2, so this is the constrained weighted least-squares
            # estimate of t.
            numerator -= inverse_variance * beta * observed
            denominator += inverse_variance * beta * beta
            count += 1
        if not count:
            raise ValueError('At least one calibration record is required to fit width.')
        squared_width = min(WIDTH_INTERVAL[1] ** 2,
                            max(WIDTH_INTERVAL[0] ** 2, numerator / denominator))
        self.width = math.sqrt(squared_width)
        return self

    def predict_calibration(self, records, width=None):
        value = self._parameter(width)
        result = []
        for row in records:
            try:
                beta = float(row['input']['beta'])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('Calibration records need a beta input.') from exc
            if not math.isfinite(beta) or not .2 <= beta <= .6:
                raise ValueError('Outside the stated calibration range.')
            result.append(-beta * value * value)
        return np.asarray(result, dtype=float)

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
