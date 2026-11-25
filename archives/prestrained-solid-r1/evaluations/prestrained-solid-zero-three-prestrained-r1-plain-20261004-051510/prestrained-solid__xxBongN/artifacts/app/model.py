"""Prediction model for the elastic bulk-wave measurements."""

import numpy as np


RHO0 = 1000.0
MODULUS_MIN = 0.8
MODULUS_MAX = 1.6


def geometry(experiment):
    """Return the deformation gradient and a unit current-space direction."""

    deformation = np.asarray(experiment["deformation"], dtype=float)
    direction = np.asarray(experiment["direction"], dtype=float)
    return deformation, direction / np.linalg.norm(direction)


def operator(experiment):
    """Return the unit-modulus current acoustic tensor and current density.

    The Hessian below is with respect to the reference deformation gradient.
    Its two propagation indices are pushed forward by ``F.T @ direction``;
    division by ``J`` converts both elasticity and density to current-volume
    quantities.
    """

    deformation, direction = geometry(experiment)
    right_cauchy_green = deformation.T @ deformation
    left_cauchy_green = deformation @ deformation.T
    identity = np.eye(3)
    determinant = np.linalg.det(deformation)
    log_determinant = np.log(determinant)
    inverse_transpose = np.linalg.inv(deformation).T
    first_invariant = np.trace(right_cauchy_green)

    # A[i, a, j, b] = d^2 w / (dF[i, a] dF[j, b]), where
    # w = W / (10^6 * modulus).
    elasticity = 0.6 * np.einsum("ij,ab->iajb", identity, identity)

    # Hessian of I2 = ((tr C)^2 - tr(C^2)) / 2.
    hessian_i2 = (
        4.0 * np.einsum("ia,jb->iajb", deformation, deformation)
        - 2.0 * np.einsum("ib,ja->iajb", deformation, deformation)
        + 2.0 * np.einsum(
            "ij,ab->iajb", first_invariant * identity - left_cauchy_green, identity
        )
        - 2.0 * np.einsum("ij,ab->iajb", identity, right_cauchy_green)
    )
    elasticity += 0.2 * hessian_i2

    # For l = log J, dl/dF[i,a] = F^{-T}[i,a] and
    # d^2l/(dF[i,a]dF[j,b]) = -F^{-T}[i,b] F^{-T}[j,a].
    log_coefficient = 2.0 * log_determinant - 1.4
    elasticity += 2.0 * np.einsum(
        "ia,jb->iajb", inverse_transpose, inverse_transpose
    )
    elasticity -= log_coefficient * np.einsum(
        "ib,ja->iajb", inverse_transpose, inverse_transpose
    )

    material_direction = deformation.T @ direction
    acoustic = np.einsum(
        "iajb,a,b->ij", elasticity, material_direction, material_direction
    ) / determinant
    acoustic = 0.5 * (acoustic + acoustic.T)
    return acoustic, RHO0 / determinant


def predict_at(experiments, modulus):
    """Predict the requested branch speed for each experiment."""

    predictions = []
    for experiment in experiments:
        acoustic, density = operator(experiment)
        eigenvalues = np.linalg.eigvalsh(acoustic)
        # Admissible inputs are strongly elliptic.  This also suppresses a
        # possible tiny negative roundoff residue before taking sqrt.
        eigenvalues = np.maximum(eigenvalues, 0.0)
        branch = int(experiment["branch"])
        predictions.append(np.sqrt(1.0e6 * modulus * eigenvalues[branch] / density))
    return np.asarray(predictions, dtype=float)


class Model:
    def __init__(self):
        self.modulus = None

    def fit(self, records):
        """Fit the shared modulus by weighted least squares in sqrt(modulus)."""

        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        basis = predict_at((record["input"] for record in records), 1.0)
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)

        weighted_basis = basis / sigma
        weighted_values = values / sigma
        sqrt_modulus = np.dot(weighted_basis, weighted_values) / np.dot(
            weighted_basis, weighted_basis
        )
        sqrt_modulus = np.clip(
            sqrt_modulus, np.sqrt(MODULUS_MIN), np.sqrt(MODULUS_MAX)
        )
        self.modulus = float(sqrt_modulus * sqrt_modulus)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.modulus)
