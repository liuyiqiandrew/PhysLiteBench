"""Wave-speed model for the finitely deformed hyperelastic solid."""

import numpy as np


RHO0 = 1000.0
_A = 0.3
_B = 0.2
_LOGJ_LINEAR = -1.4
_ENERGY_SCALE = 1.0e6


def geometry(e):
    """Return a deformation gradient and a unit current-space direction."""

    f = np.asarray(e["deformation"], dtype=float)
    n = np.asarray(e["direction"], dtype=float)
    return f, n / np.linalg.norm(n)


def _material_acoustic(f, n):
    """Return the material Hessian contracted with the reference wavevector."""

    c = f.T @ f
    b = f @ f.T
    m = f.T @ n
    j = np.linalg.det(f)

    m2 = np.dot(m, m)
    fm = f @ m
    fm2 = np.dot(fm, fm)
    i1 = np.trace(c)

    # Hessian of A*(I1 - 3) + B*(I2 - 3), contracted with m twice.
    h = (_A * 2.0 * m2) * np.eye(3)
    h += _B * (
        2.0 * (i1 * m2 - fm2) * np.eye(3)
        + 2.0 * np.outer(fm, fm)
        - 2.0 * m2 * b
    )

    # For g(log J) = -1.4 log J + (log J)^2, g' = 2 log J - 1.4.
    # Since F^{-T} m = n, its rank-one Hessian contribution is this
    # coefficient times n outer n.
    log_j = np.log(j)
    h += (2.0 - (2.0 * log_j + _LOGJ_LINEAR)) * np.outer(n, n)
    return 0.5 * (h + h.T)


def operator(e):
    """Return the current acoustic tensor and current mass density."""

    f, n = geometry(e)
    volume = np.linalg.det(f)
    material_acoustic = _material_acoustic(f, n)
    return material_acoustic / volume, RHO0 / volume


def predict_at(experiments, modulus):
    """Predict speeds for a scalar modulus in MPa."""

    result = []
    for e in experiments:
        acoustic, density = operator(e)
        branch = int(e["branch"])
        eigenvalues = np.linalg.eigvalsh(acoustic)
        # Guard only against a tiny negative round-off at a zero eigenvalue.
        squared_speed = _ENERGY_SCALE * modulus * eigenvalues[branch] / density
        result.append(np.sqrt(max(0.0, float(squared_speed))))
    return np.asarray(result, dtype=float)


class Model:
    def __init__(self):
        self.modulus = None

    def fit(self, records):
        """Fit the shared modulus by weighted least squares in speed."""

        records = list(records)
        inputs = [record["input"] for record in records]
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)

        # For fixed geometry, speed = sqrt(modulus) * speed_at_modulus_one.
        basis = predict_at(inputs, 1.0)
        weights = 1.0 / sigma
        amplitude = np.dot(basis * weights, values * weights) / np.dot(
            basis * weights, basis * weights
        )
        amplitude = np.clip(amplitude, np.sqrt(0.8), np.sqrt(1.6))
        self.modulus = float(amplitude * amplitude)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.modulus)
