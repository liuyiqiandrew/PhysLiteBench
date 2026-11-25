import numpy as np

RHO0 = 1000.0


def geometry(e):
    f = np.asarray(e['deformation'], dtype=float)
    n = np.asarray(e['direction'], dtype=float)
    return f, n / np.linalg.norm(n)


def operator(e):
    f, n = geometry(e)
    c = f.T @ f
    inverse_transpose = np.linalg.inv(f).T
    volume = np.linalg.det(f)
    coefficient = 2.0 * np.log(volume) - 1.4
    material_direction = f.T @ n
    acoustic = np.empty((3, 3))
    for j in range(3):
        change = np.outer(np.eye(3)[j], material_direction)
        dc = change.T @ f + f.T @ change
        di1 = np.trace(dc)
        dlog = np.trace(np.linalg.solve(f, change))
        dp = 0.6 * change
        dp += 0.4 * (di1 * f + np.trace(c) * change - change @ c - f @ dc)
        dp += 2.0 * dlog * inverse_transpose
        dp -= coefficient * inverse_transpose @ change.T @ inverse_transpose
        acoustic[:, j] = dp @ material_direction / volume
    return 0.5 * (acoustic + acoustic.T), RHO0 / volume


def predict_at(experiments, modulus):
    result = []
    for e in experiments:
        acoustic, density = operator(e)
        eigenvalues = np.linalg.eigvalsh(acoustic)
        result.append(np.sqrt(1e6 * modulus * eigenvalues[int(e['branch'])] / density))
    return np.asarray(result)


class Model:
    def __init__(self):
        self.modulus = None

    def fit(self, records):
        records = list(records)
        basis = predict_at([r['input'] for r in records], 1.0)
        values = np.asarray([r['value'] for r in records])
        sigma = np.asarray([r['sigma'] for r in records])
        amplitude = np.dot(basis / sigma, values / sigma) / np.dot(basis / sigma, basis / sigma)
        self.modulus = float(np.clip(amplitude, np.sqrt(0.8), np.sqrt(1.6)) ** 2)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.modulus)
