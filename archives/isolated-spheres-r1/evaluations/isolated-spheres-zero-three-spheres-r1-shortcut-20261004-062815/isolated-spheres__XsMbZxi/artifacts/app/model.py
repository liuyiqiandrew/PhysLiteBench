import numpy as np
from scipy.special import gammaln


def pair_force(separation, radius, cutoff=36):
    result = 0.0
    for m in range(cutoff + 1):
        ell = np.arange(m, cutoff + 1)
        left, right = ell[:, None], ell[None, :]
        degree = left + right + 1
        normalization = gammaln(left + right + 1) - 0.5 * (
            gammaln(left + m + 1) + gammaln(left - m + 1)
            + gammaln(right + m + 1) + gammaln(right - m + 1))
        coupling = np.exp(normalization + degree * np.log(radius / separation))
        derivative = -degree * coupling / separation
        matrix = np.eye(len(ell)) - coupling @ coupling
        result += (1 if m == 0 else 2) * np.trace(
            np.linalg.solve(matrix, coupling @ derivative))
    return float(result)


def predict_at(experiments, radius):
    result = []
    for e in experiments:
        if e['kind'] == 'dipole':
            result.append(radius**3 * e['field'])
        else:
            result.append(pair_force(e['separation'], radius))
    return np.asarray(result)


class Model:
    def __init__(self):
        self.radius = None

    def fit(self, records):
        records = list(records)
        fields = np.array([r['input']['field'] for r in records])
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        slope = np.sum(fields * values / sigma**2) / np.sum(fields**2 / sigma**2)
        self.radius = float(np.cbrt(slope))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.radius)
