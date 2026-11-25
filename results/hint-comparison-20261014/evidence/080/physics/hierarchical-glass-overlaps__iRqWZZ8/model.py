"""Equilibrium calibration and configuration coincidences."""
import math
from numbers import Real, Integral
import numpy as np

WIDTH_INTERVAL = (0.8, 1.2)
ENTROPIES = (0.3, 0.7)
VARIANCES = (0.7, 0.3)


def level_parameter(beta, width, level):
    """Return the (capped) Poisson--Dirichlet parameter at one level."""
    critical = math.sqrt(2*ENTROPIES[level]/VARIANCES[level])/width
    return min(1.0, critical/beta)


def participation(m, replicas):
    """E[sum_i p_i**replicas] for PD(m, 0) weights."""
    result = 1.0
    for j in range(1, replicas):
        result *= (j-m)/j
    return result


def leaf_partition(alpha, blocks):
    """E[prod_j p_{i_j}**blocks[j]] for distinct PD(alpha, 0) atoms.

    The block order is fixed.  Consequently this is one labelled event,
    rather than the EPPF summed over permutations of the blocks.
    """
    n = sum(blocks)
    k = len(blocks)
    result = alpha**(k-1)*math.factorial(k-1)/math.factorial(n-1)
    for size in blocks:
        for j in range(1,size):
            result *= j-alpha
    return result


def coincidence(beta, replicas, observable, width, blocks=None):
    family_parameter = level_parameter(beta, width, 0)
    descendant_parameter = level_parameter(beta, width, 1)

    # A family is selected with its descendant partition sum as a mark.  In
    # the frozen descendant phase that total is a stable random variable and
    # the family selection changes its law by a fractional
    # (family_parameter) power.  Its normalized jumps are independent of the
    # total, so the reweighted descendant event is still the PD EPPF below.
    parent = participation(family_parameter, replicas)
    if observable == 'parent_collision':
        return parent
    if observable == 'state_collision':
        return parent*participation(descendant_parameter, replicas)
    return parent*leaf_partition(descendant_parameter, blocks)


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
        # All supplied calibration temperatures are below the first critical
        # temperature for every allowed width.  Thus both levels are in their
        # Gaussian/high-temperature phase and
        #     E[E]/N = -beta * width**2.
        # The inverse-variance weighted least-squares estimate is therefore
        # linear in width**2.
        numerator = 0.0
        denominator = 0.0
        count = 0
        for row in records:
            try:
                beta = float(row['input']['beta'])
                measured = float(row['value'])
                sigma = float(row['sigma'])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError('Malformed calibration record.') from exc
            if (not math.isfinite(beta) or not .2 <= beta <= .6
                    or not math.isfinite(measured)
                    or not math.isfinite(sigma) or sigma <= 0):
                raise ValueError('Malformed calibration record.')
            weight = 1.0/(sigma*sigma)
            numerator += beta*measured*weight
            denominator += beta*beta*weight
            count += 1
        if not count or denominator <= 0:
            raise ValueError('At least one calibration record is required.')

        width_squared = -numerator/denominator
        width_squared = min(1.2**2, max(.8**2, width_squared))
        self.width = math.sqrt(width_squared)
        return self

    def predict_calibration(self, records, width=None):
        value = self._parameter(width)
        result = []
        for row in records:
            beta = float(row['input']['beta'])
            if not math.isfinite(beta) or not .2 <= beta <= .6:
                raise ValueError('Outside the stated calibration range.')
            # This is the relevant branch for the calibration data.  Keeping
            # the two level contributions explicit also documents why the
            # total variance is one here.
            energy = 0.0
            for entropy, variance in zip(ENTROPIES, VARIANCES):
                critical = math.sqrt(2*entropy/variance)/value
                if beta <= critical:
                    energy -= beta*variance*value*value
                else:
                    energy -= value*math.sqrt(2*entropy*variance)
            result.append(energy)
        return np.asarray(result,dtype=float)

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
