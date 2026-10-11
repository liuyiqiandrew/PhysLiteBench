"""Limiting osmotic coefficient for the finite two-dimensional ion cloud."""

import numpy as np
from scipy.optimize import minimize_scalar


_COUPLING_BOUNDS = (0.7, 1.4)


def coefficient(gamma, count):
    """Return the limiting osmotic coefficient for ``count`` ions.

    If ``k`` ions reach the outer wall on the scale of ``R`` while the
    remaining ions stay at finite radius, the configurational integral has
    an outer-radius exponent

        s_k = 2*k - gamma*k*(k + 1)/2.

    The pressure limit selects the largest of these exponents, including the
    k=0 (fully bound) sector.  There are only one to four ions, so retaining
    the discrete maximization is exact and avoids treating the number of
    free ions as continuous.
    """
    count = int(count)
    if count < 1:
        raise ValueError("ion count must be positive")

    k = np.arange(count + 1, dtype=float)
    exponents = 2.0 * k - float(gamma) * k * (k + 1.0) / 2.0
    return float(np.max(exponents) / (2.0 * count))


def predict_at(experiments, coupling):
    """Predict one coefficient per experiment, preserving input order."""
    return np.asarray(
        [
            coefficient(
                experiment["temperature_multiplier"] * coupling,
                experiment["ion_count"],
            )
            for experiment in experiments
        ],
        dtype=float,
    )


class Model:
    def __init__(self):
        self.coupling = None

    def fit(self,records):
        """Fit the shared reference coupling by weighted least squares."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record["input"] for record in records]
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigmas = np.asarray([record["sigma"] for record in records], dtype=float)
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(sigmas)):
            raise ValueError("calibration values and sigmas must be finite")
        if np.any(sigmas <= 0):
            raise ValueError("calibration sigmas must be positive")

        def objective(coupling):
            residual = (predict_at(experiments, coupling) - values) / sigmas
            return float(np.dot(residual, residual))

        result = minimize_scalar(
            objective,
            bounds=_COUPLING_BOUNDS,
            method="bounded",
            options={"xatol": 1e-12},
        )
        # Include the endpoints explicitly; bounded minimization does not
        # necessarily evaluate them.
        candidates = [result.x, *_COUPLING_BOUNDS]
        self.coupling = float(min(candidates, key=objective))
        return self

    def predict(self,experiments):
        if self.coupling is None:
            raise RuntimeError("fit the model before predicting")
        return predict_at(experiments,self.coupling)
