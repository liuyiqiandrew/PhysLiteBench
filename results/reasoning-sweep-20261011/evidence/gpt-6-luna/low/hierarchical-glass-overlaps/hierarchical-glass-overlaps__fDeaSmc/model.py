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
    """Infinite size quenched mean energy per degree of freedom.

    Each level is a REM with entropy s and energy variance v*N*width**2.
    Its free energy is annealed above beta_c and freezes to its ground
    state value below beta_c.  Differentiating the sum of level free
    energies gives the equilibrium energy.
    """
    energy = 0.0
    for entropy, variance in zip(ENTROPIES, VARIANCES):
        critical = math.sqrt(2*entropy/variance)/width
        if beta < critical:
            energy -= beta*variance*width*width
        else:
            energy -= math.sqrt(2*entropy*variance)*width
    return energy


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
            raise ValueError('At least one calibration record is required.')
        betas = np.asarray([float(row['input']['beta']) for row in rows])
        values = np.asarray([float(row['value']) for row in rows])
        sigmas = np.asarray([float(row['sigma']) for row in rows])
        if (not np.all(np.isfinite(betas)) or np.any((betas < .2) | (betas > .6))
                or not np.all(np.isfinite(values)) or not np.all(np.isfinite(sigmas))
                or np.any(sigmas <= 0)):
            raise ValueError('Invalid calibration record.')

        # The prediction is continuous and piecewise smooth in width.  A
        # bounded dense scan followed by local golden-section refinement
        # avoids adding a SciPy dependency and handles the transition kinks.
        lo, hi = WIDTH_INTERVAL
        grid = np.linspace(lo, hi, 4001)
        def objective(width):
            pred = np.asarray([calibration_mean(float(beta), width) for beta in betas])
            return float(np.sum(((pred-values)/sigmas)**2))
        scores = np.asarray([objective(float(w)) for w in grid])
        i = int(np.argmin(scores))
        a = float(grid[max(0, i-1)])
        b = float(grid[min(len(grid)-1, i+1)])
        ratio = (math.sqrt(5)-1)/2
        c = b-ratio*(b-a)
        d = a+ratio*(b-a)
        fc, fd = objective(c), objective(d)
        for _ in range(80):
            if fc <= fd:
                b, d, fd = d, c, fc
                c = b-ratio*(b-a)
                fc = objective(c)
            else:
                a, c, fc = c, d, fd
                d = a+ratio*(b-a)
                fd = objective(d)
        candidates = [(float(grid[i]), float(scores[i])), ((a+b)/2, objective((a+b)/2)), (lo, objective(lo)), (hi, objective(hi))]
        self.width = min(candidates, key=lambda item: item[1])[0]
        return self

    def predict_calibration(self, records, width=None):
        value = self._parameter(width)
        result = []
        for row in records:
            beta = float(row['input']['beta'])
            if not math.isfinite(beta) or not .2 <= beta <= .6:
                raise ValueError('Outside the stated calibration temperature range.')
            result.append(calibration_mean(beta, value))
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
