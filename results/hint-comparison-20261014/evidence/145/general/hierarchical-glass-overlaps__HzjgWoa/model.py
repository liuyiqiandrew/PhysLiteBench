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
        """Fit the material width by the Gaussian-error maximum likelihood rule.

        Throughout the calibration range both levels are unfrozen (even at the
        largest allowed width).  Consequently the limiting energy density is
        ``-beta * width**2``.  The likelihood is therefore a one-parameter
        weighted linear least-squares problem in ``width**2``.
        """
        betas = []
        readings = []
        sigmas = []
        for row in records:
            beta = self._calibration_beta(row)
            try:
                measured = row['value']
                sigma = row['sigma']
            except (KeyError, TypeError) as exc:
                raise ValueError('Each calibration record needs value and sigma.') from exc
            if isinstance(measured, bool) or isinstance(sigma, bool):
                raise ValueError('Calibration values and sigmas must be real.')
            try:
                measured = float(measured)
                sigma = float(sigma)
            except (TypeError, ValueError) as exc:
                raise ValueError('Calibration values and sigmas must be real.') from exc
            if not math.isfinite(measured) or not math.isfinite(sigma) or sigma <= 0:
                raise ValueError('Calibration values must be finite and sigmas positive.')
            betas.append(beta)
            readings.append(measured)
            sigmas.append(sigma)

        if not betas:
            raise ValueError('At least one calibration record is required.')

        # Scaling by the smallest sigma leaves the optimum unchanged and
        # avoids overflow for otherwise valid, extremely small uncertainties.
        smallest_sigma = min(sigmas)
        weights = [(smallest_sigma / sigma) ** 2 for sigma in sigmas]
        denominator = sum(weight * beta * beta for weight, beta in zip(weights, betas))
        estimate = -sum(weight * beta * reading
                        for weight, beta, reading in zip(weights, betas, readings)) / denominator
        estimate = min(WIDTH_INTERVAL[1] ** 2, max(WIDTH_INTERVAL[0] ** 2, estimate))
        self.width = math.sqrt(estimate)
        return self

    @staticmethod
    def _calibration_beta(row):
        try:
            beta = row['input']['beta']
        except (KeyError, TypeError) as exc:
            raise ValueError('Each calibration record needs input.beta.') from exc
        if isinstance(beta, bool):
            raise ValueError('Calibration beta must be real.')
        try:
            beta = float(beta)
        except (TypeError, ValueError) as exc:
            raise ValueError('Calibration beta must be real.') from exc
        if not math.isfinite(beta) or not .2 <= beta <= .6:
            raise ValueError('Outside the stated calibration inverse-temperature range.')
        return beta

    def predict_calibration(self, records, width=None):
        value = self._parameter(width)
        return np.asarray([-self._calibration_beta(row) * value * value for row in records],
                          dtype=float)

    def predict(self, experiments, width=None):
        value = self._parameter(width)
        result = []
        for row in experiments:
            try:
                beta = row['beta']
                replicas = row['replicas']
                observable = row['observable']
            except (KeyError, TypeError) as exc:
                raise ValueError('Each experiment needs beta, replicas, and observable.') from exc
            if isinstance(beta, bool):
                raise ValueError('Inverse temperature must be real.')
            try:
                beta = float(beta)
            except (TypeError, ValueError) as exc:
                raise ValueError('Inverse temperature must be real.') from exc
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
                except (KeyError, TypeError) as exc:
                    raise ValueError('Leaf partitions must be iterable.') from exc
                if (not blocks or any(isinstance(n,bool) or not isinstance(n,Integral) or n<1 for n in blocks)
                        or sum(blocks)!=replicas):
                    raise ValueError('Partition sizes must be positive integers summing to replicas.')
            result.append(coincidence(beta,replicas,observable,value,blocks))
        return np.asarray(result,dtype=float)
