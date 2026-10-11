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
        # In the stated calibration range both levels are in their
        # high-temperature phase.  The limiting mean energy per size is
        # therefore -beta * width**2.  Since the measurement errors are
        # independent Gaussians with known standard deviations, the
        # maximum-likelihood estimate is weighted least squares in width**2.
        numerator = 0.0
        denominator = 0.0
        count = 0
        try:
            iterator = iter(records)
        except TypeError as exc:
            raise ValueError('Calibration records must be iterable.') from exc

        for row in iterator:
            try:
                beta = row['input']['beta']
                observed = row['value']
                sigma = row['sigma']
            except (KeyError, TypeError) as exc:
                raise ValueError('Each calibration record needs input.beta, value, and sigma.') from exc
            if any(isinstance(v, bool) or not isinstance(v, Real) for v in (beta, observed, sigma)):
                raise ValueError('Calibration beta, value, and sigma must be real numbers.')
            beta, observed, sigma = float(beta), float(observed), float(sigma)
            if not math.isfinite(beta) or not .2 <= beta <= .6:
                raise ValueError('Calibration beta must be in [0.2,0.6].')
            if not math.isfinite(observed) or not math.isfinite(sigma) or sigma <= 0:
                raise ValueError('Calibration values must be finite and sigma positive.')
            weight = 1.0 / (sigma * sigma)
            numerator += weight * beta * (-observed)
            denominator += weight * beta * beta
            count += 1

        if count == 0 or denominator == 0:
            raise ValueError('At least one valid calibration record is required.')
        width_squared = numerator / denominator
        estimate = math.sqrt(min(1.2**2, max(.8**2, width_squared)))
        self.width = estimate
        return self

    def predict_calibration(self, records, width=None):
        value = self._parameter(width)
        result = []
        for row in records:
            try:
                beta = row['input']['beta']
            except (KeyError, TypeError) as exc:
                raise ValueError('Each calibration record needs input.beta.') from exc
            if isinstance(beta, bool) or not isinstance(beta, Real):
                raise ValueError('Calibration beta must be a real number.')
            beta = float(beta)
            if not math.isfinite(beta) or not .2 <= beta <= .6:
                raise ValueError('Calibration beta must be in [0.2,0.6].')
            result.append(-beta * value * value)
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
