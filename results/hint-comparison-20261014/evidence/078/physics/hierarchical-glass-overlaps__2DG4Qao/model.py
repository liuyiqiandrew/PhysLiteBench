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
    """Probability of a labelled block partition for ``PD(alpha, 0)``.

    This remains useful as the ordinary, unselected descendant result.  The
    hierarchy itself uses :func:`selected_leaf_partition` below.
    """
    n = sum(blocks)
    k = len(blocks)
    result = alpha**(k-1)*math.factorial(k-1)/math.factorial(n-1)
    for size in blocks:
        for j in range(1,size):
            result *= j-alpha
    return result


def selected_leaf_partition(q, alpha, blocks):
    """Descendant partition after selecting a family from the full Gibbs law.

    The family mass contains its descendant partition sum.  In the Poisson
    cascade limit, selecting that family tilts its descendant stable sum by
    its ``q``-th power.  Its normalized leaf weights therefore have the
    ``PD(alpha, -q)`` law.  This is important even when the requested event
    contains several distinct leaves.
    """
    n = sum(blocks)
    k = len(blocks)
    result = 1.0
    for i in range(1, k):
        result *= i * alpha - q
    for j in range(1, n):
        result /= j - q
    for size in blocks:
        for j in range(1, size):
            result *= j - alpha
    return result


def joint_leaf_partition(q, alpha, blocks):
    """Full Gibbs probability of a parent-and-leaf block event.

    This is ``participation(q, n) * selected_leaf_partition(...)`` with the
    common factors cancelled.  Besides being simpler, that form is finite at
    the high-temperature boundary ``q == 1`` where conditioning on a parent
    collision would itself be a zero-probability limiting event.
    """
    n = sum(blocks)
    result = 1.0 / math.factorial(n - 1)
    for i in range(1, len(blocks)):
        result *= i * alpha - q
    for size in blocks:
        for j in range(1, size):
            result *= j - alpha
    return result


def coincidence(beta, replicas, observable, width, blocks=None):
    q = level_parameter(beta,width,0)
    alpha = level_parameter(beta,width,1)
    parent = participation(q,replicas)
    if observable == 'parent_collision':
        return parent
    if observable == 'state_collision':
        # The two factors below cancel the selection tilt: globally the leaf
        # weights are PD(alpha, 0), so this is not parent * participation.
        return participation(alpha,replicas)
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
        # In the supplied calibration range both levels are annealed, hence
        # the limiting energy density is -beta * width**2.  Independent
        # Gaussian reading errors make weighted least squares the MLE.
        numerator = 0.0
        denominator = 0.0
        count = 0
        for row in records:
            try:
                beta = float(row['input']['beta'])
                energy = float(row['value'])
                sigma = float(row['sigma'])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError('Calibration records need finite beta, value, and sigma.') from error
            if (not math.isfinite(beta) or not .2 <= beta <= .6
                    or not math.isfinite(energy) or not math.isfinite(sigma)
                    or sigma <= 0.0):
                raise ValueError('Calibration records need beta in [0.2, 0.6] and positive finite sigma.')
            weight = 1.0/(sigma*sigma)
            numerator -= weight*beta*energy
            denominator += weight*beta*beta
            count += 1
        if count == 0:
            raise ValueError('At least one calibration record is required to fit width.')
        squared = min(WIDTH_INTERVAL[1]**2, max(WIDTH_INTERVAL[0]**2, numerator/denominator))
        self.width = math.sqrt(squared)
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
