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


def leaf_partition(parent_alpha, leaf_alpha, blocks):
    """Probability of one labelled leaf partition with a common parent.

    The leaf weights within a sampled parent have a PD(leaf_alpha,
    -parent_alpha) law.  Multiplying its sampling formula by the
    probability of a common parent cancels the parent denominators.
    """
    n = sum(blocks)
    result = 1.0/math.factorial(n-1)
    for j in range(1, len(blocks)):
        result *= j*leaf_alpha-parent_alpha
    for size in blocks:
        for j in range(1, size):
            result *= j-leaf_alpha
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
        observations = []
        for row in records:
            beta = calibration_beta(row)
            measured = float(row['value'])
            sigma = float(row['sigma'])
            if not math.isfinite(measured) or not math.isfinite(sigma) or sigma <= 0:
                raise ValueError('Calibration values must be finite and sigma positive.')
            observations.append((beta, measured, sigma))
        if not observations:
            raise ValueError('At least one calibration record is required to fit width.')

        # In the calibration range both levels are unfrozen, so E/N=-beta*width^2.
        # Gaussian maximum likelihood is weighted least squares in width^2.
        sigma_scale = min(sigma for _, _, sigma in observations)
        numerator = math.fsum(-beta*measured*(sigma_scale/sigma)**2
                              for beta, measured, sigma in observations)
        denominator = math.fsum(beta*beta*(sigma_scale/sigma)**2
                                for beta, _, sigma in observations)
        width_squared = min(max(numerator/denominator, WIDTH_INTERVAL[0]**2),
                            WIDTH_INTERVAL[1]**2)
        self.width = math.sqrt(width_squared)
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
