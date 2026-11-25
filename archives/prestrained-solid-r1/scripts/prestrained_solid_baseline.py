import numpy as np

RHO0 = 1000.0


def geometry(e):
    f = np.asarray(e['deformation'], dtype=float)
    n = np.asarray(e['direction'], dtype=float)
    return f, n / np.linalg.norm(n)


def operator(e):
    f, n = geometry(e)
    c = f.T @ f
    ci = np.linalg.inv(c)
    eye = np.eye(3)
    volume = np.linalg.det(f)
    coefficient = 2.0 * np.log(volume) - 1.4
    tensor = 0.8 * (np.einsum('ab,cd->abcd', eye, eye)
                   - 0.5 * (np.einsum('ac,bd->abcd', eye, eye)
                            + np.einsum('ad,bc->abcd', eye, eye)))
    tensor += 2.0 * np.einsum('ab,cd->abcd', ci, ci)
    tensor -= coefficient * (np.einsum('ac,bd->abcd', ci, ci)
                            + np.einsum('ad,bc->abcd', ci, ci))
    material_direction = f.T @ n
    acoustic = np.einsum('ib,a,bacd,c,jd->ij', f, material_direction,
                         tensor, material_direction, f) / volume
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
