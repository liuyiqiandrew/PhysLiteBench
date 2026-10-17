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


def calibration_mean(beta, width):
    """Disorder-averaged limiting mean energy per size.

    Each Gaussian level contributes its REM energy density.  The records in
    the calibration set have beta <= .6, while even the smallest critical
    inverse temperature over the allowed width interval is greater than .6,
    so both levels are in their annealed regimes for those records.  Keeping
    the level factors here also makes the expression faithful to the model if
    it is used outside that calibration range.
    """
    q = level_parameter(beta, width, 0)
    alpha = level_parameter(beta, width, 1)
    return -beta * width**2 * (VARIANCES[0] * q + VARIANCES[1] * alpha)


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
        # In the stated calibration range q=alpha=1 for every width in the
        # allowed interval, hence the model is linear in t=width**2:
        #     mean(beta) = -beta*t.
        # The sigma values are standard deviations of independent readings,
        # so inverse-variance weighted least squares is the appropriate fit.
        beta_values = []
        measured_values = []
        inverse_variances = []
        for row in records:
            try:
                beta = float(row['input']['beta'])
                measured = float(row['value'])
                sigma = float(row['sigma'])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('Malformed calibration record.') from exc
            if (not math.isfinite(beta) or not math.isfinite(measured)
                    or not math.isfinite(sigma) or sigma <= 0):
                raise ValueError('Calibration records must contain finite values and positive sigma.')
            if not .2 <= beta <= .6:
                raise ValueError('Calibration beta must be in [0.2,0.6].')
            beta_values.append(beta)
            measured_values.append(measured)
            inverse_variances.append(1.0 / (sigma * sigma))

        if not beta_values:
            raise ValueError('At least one calibration record is required.')

        beta_values = np.asarray(beta_values, dtype=float)
        measured_values = np.asarray(measured_values, dtype=float)
        inverse_variances = np.asarray(inverse_variances, dtype=float)
        denominator = np.sum(inverse_variances * beta_values**2)
        squared_width = -np.sum(inverse_variances * beta_values * measured_values) / denominator

        # The material scale is known to lie in the stated interval.  A noisy
        # calibration can therefore only move the constrained estimate to an
        # endpoint, rather than leaving an invalid fitted model.
        squared_width = float(np.clip(squared_width, WIDTH_INTERVAL[0]**2,
                                      WIDTH_INTERVAL[1]**2))
        self.width = math.sqrt(squared_width)
        return self

    def predict_calibration(self, records, width=None):
        value = self._parameter(width)
        return np.asarray([calibration_mean(float(row['input']['beta']), value)
                           for row in records], dtype=float)

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
