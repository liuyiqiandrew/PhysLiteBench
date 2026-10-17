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


def leaf_partition(alpha, blocks, theta=0.0):
    """One labelled partition event under a PD(alpha, theta) paintbox."""
    n = sum(blocks)
    k = len(blocks)
    numerator = math.prod(theta + j*alpha for j in range(1, k))
    denominator = math.prod(theta + j for j in range(1, n))
    result = numerator/denominator
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
        blocks = [replicas]
    if parent == 0.0:
        return 0.0
    # Conditioning on a family selected by its Gibbs mass tilts its child
    # partition by the q-th power of its stable total mass.  The normalized
    # child weights therefore have PD(alpha, -q), rather than PD(alpha, 0).
    return parent*leaf_partition(alpha,blocks,theta=-q)


class Model:
    def __init__(self):
        self.width = None

    @staticmethod
    def _default_records():
        path = Path(__file__).resolve().parent / 'data' / 'calibration.json'
        with path.open(encoding='utf-8') as stream:
            return json.load(stream)

    def _parameter(self, width):
        if width is not None and (isinstance(width,bool) or not isinstance(width,Real)):
            raise ValueError('Supply a real material scale.')
        if width is None and self.width is None:
            self.fit(self._default_records())
        value = self.width if width is None else float(width)
        if value is None or not math.isfinite(value) or not .8 <= value <= 1.2:
            raise ValueError('Fit or supply width in [0.8,1.2].')
        return value

    def fit(self, records):
        try:
            rows = list(records)
        except TypeError as exc:
            raise ValueError('Calibration records must be iterable.') from exc
        if not rows:
            raise ValueError('At least one calibration record is required.')

        betas, observations, sigmas = [], [], []
        for row in rows:
            try:
                beta = float(row['input']['beta'])
                observed = float(row['value'])
                sigma = float(row['sigma'])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError('Malformed calibration record.') from exc
            if (not all(map(math.isfinite, (beta, observed, sigma)))
                    or not .2 <= beta <= .6 or sigma <= 0):
                raise ValueError('Calibration values must be finite, with beta in [0.2,0.6] and sigma positive.')
            betas.append(beta)
            observations.append(observed)
            sigmas.append(sigma)

        # In this temperature range both levels are unfrozen, so the limiting
        # mean energy per size is -beta*width**2.  Weighted least squares is
        # linear in width**2, with the supplied measurement variances.
        sigma_scale = min(sigmas)
        weights = [(sigma_scale/sigma)**2 for sigma in sigmas]
        numerator = math.fsum(w*beta*value
                              for w, beta, value in zip(weights, betas, observations))
        denominator = math.fsum(w*beta*beta
                                for w, beta in zip(weights, betas))
        width_squared = -numerator/denominator
        lo, hi = WIDTH_INTERVAL
        self.width = math.sqrt(min(hi*hi, max(lo*lo, width_squared)))
        return self

    def predict_calibration(self, records, width=None):
        value = self._parameter(width)
        result = []
        for row in records:
            beta = float(row['input']['beta'])
            if not math.isfinite(beta) or not .2 <= beta <= .6:
                raise ValueError('Outside the stated calibration inverse-temperature range.')
            result.append(-beta*value*value)
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
