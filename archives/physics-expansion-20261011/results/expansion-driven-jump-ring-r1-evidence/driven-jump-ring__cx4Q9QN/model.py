"""Model for the driven three-state molecular ring.

The matrices in this module act on column vectors of state probabilities. A
tilted off-diagonal entry is the rate of a jump multiplied by the exponential
of the corresponding counter increment. Repeating a period therefore means
raising the period propagator to a power; the long-run scaled cumulant
generating function is the logarithm of its Perron eigenvalue.
"""

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar


def generator(stage, kappa, chi):
    """Return the tilted generator for one stage."""

    energies = np.asarray(stage["energies"], dtype=float)
    barriers = np.asarray(stage["barriers"], dtype=float)
    affinity = float(stage["affinity"])
    kappa = float(kappa)
    chi = float(chi)

    matrix = np.zeros((3, 3), dtype=float)
    for i in range(3):
        j = (i + 1) % 3
        clockwise = kappa * np.exp(
            energies[i] - barriers[i] + affinity / 6.0
        )
        reverse = kappa * np.exp(
            energies[j] - barriers[i] - affinity / 6.0
        )

        # Columns are source states and rows are destination states.
        matrix[j, i] += clockwise * np.exp(chi / 3.0)
        matrix[i, j] += reverse * np.exp(-chi / 3.0)
        matrix[i, i] -= clockwise
        matrix[j, j] -= reverse

    return matrix


def cycle(experiment, kappa, chi):
    """Return the tilted propagator for one complete period."""

    propagator = np.eye(3)
    for stage in experiment["stages"]:
        # Stages occur in list order, so the latest stage multiplies on the
        # left when acting on a column state vector.
        propagator = expm(
            float(stage["duration"]) * generator(stage, kappa, chi)
        ) @ propagator
    return propagator


def _log_spectral_radius(matrix):
    """Return the log of the spectral radius of a positive 3x3 matrix."""

    eigenvalues = np.linalg.eigvals(matrix)
    radius = float(np.max(np.abs(eigenvalues)))
    return float(np.log(radius))


def response(experiment, kappa):
    """Evaluate the asymptotic log moment generating function per period."""

    chi = float(experiment["field"])
    if chi == 0.0:
        # At zero field the propagator is stochastic and the answer is
        # exactly zero; this also avoids reporting round-off at this useful
        # normalization point.
        return 0.0
    return _log_spectral_radius(cycle(experiment, kappa, chi))


def predict_at(experiments, attempt_rate):
    """Predict the response for each experiment in input order."""

    return np.asarray(
        [response(experiment, float(attempt_rate)) for experiment in experiments],
        dtype=float,
    )


class Model:
    def __init__(self):
        self.attempt_rate = None

    def fit(self, records):
        """Estimate the common attempt rate by weighted least squares."""

        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record["input"] for record in records]
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigmas = np.asarray([record["sigma"] for record in records], dtype=float)
        if np.any(~np.isfinite(values)) or np.any(~np.isfinite(sigmas)):
            raise ValueError("calibration values and sigmas must be finite")
        if np.any(sigmas <= 0.0):
            raise ValueError("calibration sigmas must be positive")

        def objective(rate):
            residual = (predict_at(experiments, rate) - values) / sigmas
            return float(np.dot(residual, residual))

        lower, upper = 0.8, 1.3
        result = minimize_scalar(
            objective,
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 1e-12},
        )

        # Include endpoints explicitly because bounded minimization is aimed
        # at an interior solution and need not return an exact endpoint.
        candidates = [(float(result.fun), float(result.x))]
        candidates.extend((objective(rate), rate) for rate in (lower, upper))
        _, best_rate = min(candidates, key=lambda item: item[0])
        self.attempt_rate = float(best_rate)
        return self

    def predict(self, experiments):
        if self.attempt_rate is None:
            raise RuntimeError("fit the model before calling predict")
        return predict_at(experiments, self.attempt_rate)
