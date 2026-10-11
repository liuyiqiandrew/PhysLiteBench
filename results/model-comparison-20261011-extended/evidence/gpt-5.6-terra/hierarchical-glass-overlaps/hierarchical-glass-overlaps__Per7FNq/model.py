"""Equilibrium calibration and configuration coincidences."""
import math
from numbers import Real, Integral
import numpy as np

WIDTH_INTERVAL = (0.8, 1.2)
ENTROPIES = (0.3, 0.7)
VARIANCES = (0.7, 0.3)


def level_parameter(beta, width, level):
    """Poisson--Dirichlet parameter of one frozen REM level.

    A value of one denotes an unfrozen level.  Keeping that endpoint is
    useful: the participation formula then correctly gives zero for a
    coincidence at an unfrozen level.
    """
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
        """Fit ``width`` by the Gaussian-error maximum likelihood estimate.

        In the supplied calibration temperature range neither REM level has
        frozen, so the limiting energy density is ``-beta * width**2``.  The
        likelihood is consequently a weighted least-squares problem in
        ``width**2`` and has a closed form solution.
        """
        numerator = 0.0
        denominator = 0.0
        count = 0
        for row in records:
            beta, measured, sigma = self._calibration_record(row, need_measurement=True)
            weight = 1.0 / (sigma * sigma)
            numerator -= beta * measured * weight
            denominator += beta * beta * weight
            count += 1
        if count == 0 or not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0.0:
            raise ValueError('Supply at least one valid calibration record.')

        squared = min(WIDTH_INTERVAL[1] ** 2,
                      max(WIDTH_INTERVAL[0] ** 2, numerator / denominator))
        self.width = math.sqrt(squared)
        return self

    @staticmethod
    def _real(value, message):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(message)
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(message)
        return value

    @classmethod
    def _calibration_record(cls, row, need_measurement=False):
        try:
            beta = cls._real(row['input']['beta'], 'Invalid calibration inverse temperature.')
            if not .2 <= beta <= .6:
                raise ValueError('Outside the stated calibration inverse-temperature range.')
            if not need_measurement:
                return beta
            measured = cls._real(row['value'], 'Invalid calibration measurement.')
            sigma = cls._real(row['sigma'], 'Invalid calibration uncertainty.')
        except (KeyError, TypeError):
            raise ValueError('Invalid calibration record.') from None
        if sigma <= 0.0:
            raise ValueError('Calibration uncertainty must be positive.')
        return beta, measured, sigma

    def predict_calibration(self, records, width=None):
        value = self._parameter(width)
        return np.asarray([-self._calibration_record(row)*value*value for row in records], dtype=float)

    def predict(self, experiments, width=None):
        value = self._parameter(width)
        result = []
        for row in experiments:
            try:
                beta = self._real(row['beta'], 'Invalid inverse temperature.')
                replicas = row['replicas']
                observable = row['observable']
            except (KeyError, TypeError):
                raise ValueError('Invalid experiment.') from None
            if not math.isfinite(beta) or not .2 <= beta <= 6:
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
