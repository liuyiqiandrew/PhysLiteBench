"""Infinite-size, disorder-averaged probabilities for the two-level GREM."""
import math
from numbers import Real, Integral
import numpy as np

WIDTH_INTERVAL = (0.8, 1.2)
ENTROPIES = (0.3, 0.7)
VARIANCES = (0.7, 0.3)


def level_parameter(beta, width, level):
    """Freezing parameter, with one denoting a diffuse Gibbs measure."""
    critical = math.sqrt(2*ENTROPIES[level]/VARIANCES[level])/width
    return min(1.0, critical/beta)


def participation(m, replicas):
    result = 1.0
    for j in range(1, replicas):
        result *= (j-m)/j
    return result


def leaf_partition(q, alpha, blocks):
    """Joint probability of one parent and a specified labelled leaf partition.

    Parent masses have law PD(q, 0). In the frozen phase their descendants
    have law PD(alpha, -q), because weighting a family by its partition
    function tilts the descendant distribution. Multiplying its partition
    probability by the parent coincidence cancels the factors (j - q).
    The resulting product also covers diffuse leaves (alpha = 1) and
    diffuse parents (q = alpha = 1), without singular gamma functions.
    """
    n = sum(blocks)
    k = len(blocks)
    result = 1.0/math.factorial(n-1)
    for j in range(1, k):
        result *= j*alpha-q
    for size in blocks:
        for j in range(1,size):
            result *= j-alpha
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
        """Fit the Gaussian measurement likelihood, constrained to the interval.

        Every calibration temperature is above both freezing temperatures,
        so the limiting energy per size is -beta * width**2. The likelihood
        is therefore a weighted least-squares problem linear in width**2.
        """
        data = []
        for row in records:
            beta = calibration_beta(row)
            measured = float(row['value'])
            sigma = float(row['sigma'])
            if not math.isfinite(measured):
                raise ValueError('Calibration values must be finite.')
            if not math.isfinite(sigma) or sigma <= 0:
                raise ValueError('Calibration sigma must be positive and finite.')
            data.append((beta, measured, sigma))
        if not data:
            raise ValueError('Supply at least one calibration record.')

        # A common weight scale cancels; avoid squaring inverse tiny sigmas.
        beta, measured, sigma = np.asarray(data, dtype=float).T
        weights = (np.min(sigma)/sigma)**2
        squared_width = -np.sum(weights*beta*measured)/np.sum(weights*beta**2)
        lower, upper = WIDTH_INTERVAL
        self.width = math.sqrt(float(np.clip(squared_width, lower**2, upper**2)))
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
