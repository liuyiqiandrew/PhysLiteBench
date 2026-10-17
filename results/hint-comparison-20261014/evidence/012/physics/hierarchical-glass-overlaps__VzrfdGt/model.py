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


def selected_leaf_partition(alpha, parent_parameter, blocks):
    """Descendant EPPF after selecting a family from the full Gibbs law.

    In the frozen descendant REM the total descendant partition sum is an
    alpha-stable variable.  A family selected through its full Gibbs weight
    is therefore tilted by the q-th power of that variable, where q is the
    parent-level parameter.  The resulting Poisson--Dirichlet EPPF is
    PD(alpha, -q).
    """
    n = sum(blocks)
    k = len(blocks)

    result = 1.0
    for j in range(1, k):
        result *= j*alpha - parent_parameter
    for j in range(1, n):
        result /= j - parent_parameter
    for size in blocks:
        for j in range(1, size):
            result *= j - alpha
    return result


def coincidence(beta, replicas, observable, width, blocks=None):
    q = level_parameter(beta,width,0)
    alpha = level_parameter(beta,width,1)
    parent = participation(q,replicas)
    if observable == 'parent_collision':
        return parent
    if parent == 0.0:
        return 0.0
    if observable == 'state_collision':
        blocks = [replicas]
    return parent*selected_leaf_partition(alpha,q,blocks)


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
        rows = list(records)
        if not rows:
            raise ValueError('Calibration records are required to fit width.')

        # Throughout the calibration range both levels are in the Gaussian
        # phase, so the limiting mean energy per size is -beta*width**2.
        beta = []
        measured = []
        sigma = []
        for row in rows:
            b = float(row['input']['beta'])
            y = float(row['value'])
            s = float(row['sigma'])
            if not math.isfinite(b) or not .2 <= b <= .6:
                raise ValueError('Calibration beta is outside [0.2,0.6].')
            if not math.isfinite(y) or not math.isfinite(s) or s <= 0:
                raise ValueError('Calibration values and sigma must be finite, with sigma positive.')
            beta.append(b)
            measured.append(y)
            sigma.append(s)

        beta = np.asarray(beta, dtype=float)
        measured = np.asarray(measured, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        design = -beta
        weights = 1.0/(sigma*sigma)
        width_squared = np.sum(weights*design*measured)/np.sum(weights*design*design)
        width_squared = min(1.2**2, max(.8**2, float(width_squared)))
        self.width = math.sqrt(width_squared)
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
