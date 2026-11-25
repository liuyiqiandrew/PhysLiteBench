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
        """Fit the material scale from the asymptotic calibration readings.

        Every calibration beta in the supplied range is below the first
        transition for every allowed width.  Consequently the limiting
        equilibrium energy density is ``-beta * width**2``.  The measured
        values have known, independent Gaussian errors, so the maximum
        likelihood estimate is a weighted least-squares estimate of
        ``width**2``.
        """
        beta_values = []
        measured = []
        sigmas = []
        for row in records:
            try:
                beta = float(row['input']['beta'])
                value = float(row['value'])
                sigma = float(row['sigma'])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError('Invalid calibration record.') from exc
            if (not math.isfinite(beta) or not .2 <= beta <= .6
                    or not math.isfinite(value)
                    or not math.isfinite(sigma) or sigma <= 0):
                raise ValueError('Invalid calibration record.')
            beta_values.append(beta)
            measured.append(value)
            sigmas.append(sigma)

        if not beta_values:
            raise ValueError('At least one calibration record is required.')

        beta_values = np.asarray(beta_values, dtype=float)
        measured = np.asarray(measured, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float)
        weights = 1.0 / (sigmas * sigmas)
        squared_width = np.sum(weights * beta_values * (-measured)) / np.sum(
            weights * beta_values * beta_values
        )
        squared_width = float(np.clip(squared_width,
                                      WIDTH_INTERVAL[0] ** 2,
                                      WIDTH_INTERVAL[1] ** 2))
        self.width = math.sqrt(squared_width)
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
