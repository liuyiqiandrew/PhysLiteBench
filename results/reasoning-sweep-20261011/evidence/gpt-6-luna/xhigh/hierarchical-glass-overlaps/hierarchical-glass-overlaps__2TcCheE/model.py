"""Equilibrium calibration and configuration coincidences."""
import json
import math
from numbers import Real, Integral
from pathlib import Path
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
        if width is not None:
            if isinstance(width,bool) or not isinstance(width,Real):
                raise ValueError('Supply a real material scale.')
            value = float(width)
        else:
            if self.width is None:
                self.fit(self._calibration_records())
            value = self.width
        if value is None or not math.isfinite(value) or not .8 <= value <= 1.2:
            raise ValueError('Fit or supply width in [0.8,1.2].')
        return value

    @staticmethod
    def _calibration_records():
        path = Path(__file__).resolve().parent / 'data' / 'calibration.json'
        with path.open(encoding='utf-8') as stream:
            return json.load(stream)

    def fit(self, records):
        """Fit the material scale by weighted least squares.

        Throughout the calibration interval both hierarchical levels are in
        their high-temperature phase, so the limiting mean energy per size is
        ``-beta * width**2``.  The regression is linear in ``width**2``.
        """
        rows = list(records)
        if not rows:
            raise ValueError('At least one calibration record is required.')

        parsed = []
        for row in rows:
            try:
                beta_value = row['input']['beta']
                energy_value = row['value']
                sigma_value = row['sigma']
                if (isinstance(beta_value, bool) or not isinstance(beta_value, Real)
                        or isinstance(energy_value, bool) or not isinstance(energy_value, Real)
                        or isinstance(sigma_value, bool) or not isinstance(sigma_value, Real)):
                    raise ValueError
                beta = float(beta_value)
                energy = float(energy_value)
                sigma = float(sigma_value)
            except (KeyError, TypeError, ValueError, OverflowError):
                raise ValueError('Each calibration record needs real beta, value, and sigma fields.') from None
            if (not math.isfinite(beta) or not .2 <= beta <= .6
                    or not math.isfinite(energy) or not math.isfinite(sigma) or sigma <= 0):
                raise ValueError('Calibration inputs must be finite and within the stated ranges.')
            parsed.append((beta, energy, sigma))

        # Rescale the weights to keep very small but valid sigmas from
        # overflowing; a common factor does not change the least-squares fit.
        smallest_sigma = min(sigma for _, _, sigma in parsed)
        terms = [((smallest_sigma / sigma) ** 2, beta, energy)
                 for beta, energy, sigma in parsed]
        denominator = math.fsum(weight * beta * beta for weight, beta, _ in terms)
        numerator = -math.fsum(weight * beta * energy for weight, beta, energy in terms)
        if not math.isfinite(denominator) or denominator <= 0 or not math.isfinite(numerator):
            raise ValueError('Calibration records do not define a finite material-scale fit.')
        width_squared = numerator / denominator
        width_squared = min(WIDTH_INTERVAL[1] ** 2,
                            max(WIDTH_INTERVAL[0] ** 2, width_squared))
        self.width = math.sqrt(width_squared)
        return self

    def predict_calibration(self, records, width=None):
        value = self._parameter(width)
        result = []
        for row in records:
            try:
                beta_value = row['input']['beta']
                if isinstance(beta_value, bool) or not isinstance(beta_value, Real):
                    raise ValueError
                beta = float(beta_value)
            except (KeyError, TypeError, ValueError, OverflowError):
                raise ValueError('Each calibration input needs a real beta.') from None
            if not math.isfinite(beta) or not .2 <= beta <= .6:
                raise ValueError('Calibration beta must be in [0.2,0.6].')
            result.append(-beta * value * value)
        return np.asarray(result, dtype=float)

    def predict(self, experiments, width=None):
        value = self._parameter(width)
        result = []
        for row in experiments:
            beta_value = row['beta']
            if isinstance(beta_value, bool) or not isinstance(beta_value, Real):
                raise ValueError('Supply a real inverse temperature.')
            beta = float(beta_value)
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
