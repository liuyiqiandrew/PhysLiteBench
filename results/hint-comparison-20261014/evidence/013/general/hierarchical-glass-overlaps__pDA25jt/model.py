"""Equilibrium calibration and configuration coincidences."""
import math
from numbers import Real, Integral
import numpy as np

WIDTH_INTERVAL = (0.8, 1.2)
ENTROPIES = (0.3, 0.7)
VARIANCES = (0.7, 0.3)


def level_parameter(beta, width, level):
    """Return the Poisson--Dirichlet parameter for one hierarchy level.

    A value of one denotes the unfrozen phase.  The two levels have their
    own entropy and variance densities, so their freezing temperatures are
    different even though they share the same material width.
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
        try:
            value = self.width if width is None else float(width)
        except (OverflowError, ValueError):
            raise ValueError('Fit or supply width in [0.8,1.2].') from None
        if value is None or not math.isfinite(value) or not .8 <= value <= 1.2:
            raise ValueError('Fit or supply width in [0.8,1.2].')
        return value

    @staticmethod
    def _real(value, message):
        """Coerce one finite, non-Boolean real value or raise ValueError."""
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(message)
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(message)
        return value

    @classmethod
    def _calibration_row(cls, row):
        """Extract and validate a calibration observation.

        All supplied calibration temperatures lie below either freezing
        transition, where the limiting mean energy per size is
        ``-beta * width**2``.
        """
        try:
            beta = cls._real(row['input']['beta'], 'Outside the stated inverse-temperature range.')
            observed = cls._real(row['value'], 'Calibration values must be finite real numbers.')
            sigma = cls._real(row['sigma'], 'Calibration sigmas must be positive finite real numbers.')
        except (KeyError, TypeError):
            raise ValueError('Each calibration record needs input, value, and sigma fields.') from None
        if not .2 <= beta <= .6:
            raise ValueError('Outside the stated inverse-temperature range.')
        if sigma <= 0:
            raise ValueError('Calibration sigmas must be positive finite real numbers.')
        return beta, observed, sigma

    def fit(self, records):
        """Maximum-likelihood fit for the common material width.

        With independent Gaussian errors, fitting ``t = width**2`` is a
        weighted zero-intercept linear regression of ``-value`` on beta.
        The physical interval is imposed after the unconstrained optimum.
        """
        numerator = 0.0
        denominator = 0.0
        count = 0
        try:
            iterator = iter(records)
        except TypeError:
            raise ValueError('Calibration records must be iterable.') from None
        for row in iterator:
            beta, observed, sigma = self._calibration_row(row)
            weight = 1.0/(sigma*sigma)
            numerator += -weight*beta*observed
            denominator += weight*beta*beta
            count += 1
        if count == 0 or not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
            raise ValueError('At least one valid calibration record is required.')
        squared_width = numerator/denominator
        squared_width = min(WIDTH_INTERVAL[1]**2,
                            max(WIDTH_INTERVAL[0]**2, squared_width))
        self.width = math.sqrt(squared_width)
        return self

    def predict_calibration(self, records, width=None):
        value = self._parameter(width)
        result = []
        try:
            iterator = iter(records)
        except TypeError:
            raise ValueError('Calibration records must be iterable.') from None
        for row in iterator:
            beta, _, _ = self._calibration_row(row)
            result.append(-beta*value*value)
        return np.asarray(result, dtype=float)

    def predict(self, experiments, width=None):
        value = self._parameter(width)
        result = []
        for row in experiments:
            try:
                beta = self._real(row['beta'], 'Outside the stated inverse-temperature range.')
                replicas = row['replicas']
                observable = row['observable']
            except (KeyError, TypeError):
                raise ValueError('Each experiment needs beta, replicas, and observable fields.') from None
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
