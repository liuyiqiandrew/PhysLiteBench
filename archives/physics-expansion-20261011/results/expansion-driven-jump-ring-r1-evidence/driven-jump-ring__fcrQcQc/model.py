"""Model for the periodically driven three-state molecular ring.

The probability vectors in this module are column vectors. Consequently,
the propagator for a stage is multiplied on the left, and the tilted
one-period propagator is the product of the stage propagators in
chronological order.
"""

import numpy as np
from scipy.linalg import eigvals, expm
from scipy.optimize import minimize_scalar


def generator(stage, kappa, chi):
    """Return the tilted continuous-time generator for one stage."""
    energy = np.asarray(stage["energies"], dtype=float)
    barrier = np.asarray(stage["barriers"], dtype=float)
    affinity = float(stage["affinity"])
    kappa = float(kappa)
    chi = float(chi)

    matrix = np.zeros((3, 3), dtype=float)
    clockwise_factor = np.exp(chi / 3.0)
    counterclockwise_factor = np.exp(-chi / 3.0)

    for i in range(3):
        j = (i + 1) % 3
        forward = kappa * np.exp(energy[i] - barrier[i] + affinity / 6.0)
        backward = kappa * np.exp(energy[j] - barrier[i] - affinity / 6.0)

        # Matrix entries use destination, source indexing.
        matrix[j, i] += forward * clockwise_factor
        matrix[i, j] += backward * counterclockwise_factor
        matrix[i, i] -= forward
        matrix[j, j] -= backward

    return matrix


def cycle(experiment, kappa, chi):
    """Return the tilted propagator over one complete period."""
    result = np.eye(3)
    for stage in experiment["stages"]:
        result = expm(float(stage["duration"]) * generator(stage, kappa, chi)) @ result
    return result


def response(experiment, kappa):
    """Return the asymptotic tilted cumulant generating function per period.

    If ``C`` is the tilted propagator for one period, the moment after N
    periods is proportional to ``C**N``. All ordinary transition rates are
    positive, so the dominant eigenvalue of ``C`` gives the growth factor.
    """
    tilted = cycle(experiment, kappa, float(experiment["field"]))
    eigenvalues = eigvals(tilted)

    # The tilted propagator is positive for these experiments. Any imaginary
    # part returned by eigvals is numerical noise.
    dominant = float(np.max(eigenvalues.real))
    if not np.isfinite(dominant) or dominant <= 0.0:
        raise FloatingPointError("invalid dominant eigenvalue")
    return float(np.log(dominant))


def predict_at(experiments, attempt_rate):
    """Evaluate a collection of experiments at a specified attempt rate."""
    return np.asarray(
        [response(experiment, float(attempt_rate)) for experiment in experiments],
        dtype=float,
    )


class Model:
    def __init__(self):
        self.attempt_rate = None

    def fit(self, records):
        """Estimate the common attempt rate from calibration observations."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record["input"] for record in records]
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigmas = np.asarray([record["sigma"] for record in records], dtype=float)
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(sigmas)):
            raise ValueError("calibration values and sigmas must be finite")
        if np.any(sigmas <= 0.0):
            raise ValueError("calibration sigmas must be positive")

        def objective(attempt_rate):
            residual = (predict_at(experiments, attempt_rate) - values) / sigmas
            return float(np.dot(residual, residual))

        lower, upper = 0.8, 1.3
        result = minimize_scalar(
            objective,
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 1e-12, "maxiter": 200},
        )

        # Include the bounds explicitly since bounded minimization does not
        # necessarily evaluate them, and the physical interval is closed.
        candidates = [(float(result.x), float(result.fun))]
        candidates.extend((bound, objective(bound)) for bound in (lower, upper))
        self.attempt_rate = min(candidates, key=lambda item: item[1])[0]
        return self

    def predict(self, experiments):
        if self.attempt_rate is None:
            raise RuntimeError("fit must be called before predict")
        return predict_at(experiments, self.attempt_rate)
