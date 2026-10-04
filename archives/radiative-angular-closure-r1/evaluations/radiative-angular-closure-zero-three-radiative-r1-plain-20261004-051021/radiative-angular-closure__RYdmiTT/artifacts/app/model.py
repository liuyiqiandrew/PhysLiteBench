import numpy as np
from scipy.optimize import minimize_scalar


def transport(experiment, absorption=0.0):
    """Return the weak-response signal for one experiment.

    A beam travelling in direction ``n`` samples the preparation at
    ``x - n_x * time``.  Differentiating the preparation with respect to the
    modulation parameter therefore gives the cosine phase below.  Absorption
    contributes the same survival factor to every beam.
    """
    directions = np.asarray(experiment["directions"], dtype=float)
    weights = np.asarray(experiment["weights"], dtype=float)
    modulations = np.asarray(experiment["modulations"], dtype=float)
    wavenumber = float(experiment["wavenumber"])
    time = float(experiment["time"])
    position = float(experiment["position"])

    phases = wavenumber * (position - directions[:, 0] * time)
    response = np.sum(weights * modulations * np.cos(phases))
    return float(np.exp(-float(absorption) * time) * response)


class Model:
    def __init__(self):
        self.absorption = .25

    def fit(self, records):
        """Fit the common absorption rate using the calibration records."""
        records = list(records)
        if not records:
            return self

        # The calibration uncertainty is independent of the rate, so the
        # Gaussian maximum-likelihood estimate is weighted least squares.
        signals = np.asarray(
            [transport(record["input"]) for record in records], dtype=float
        )
        observed = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)
        times = np.asarray(
            [float(record["input"]["time"]) for record in records], dtype=float
        )

        def objective(absorption):
            residual = (signals * np.exp(-absorption * times) - observed) / sigma
            return float(np.dot(residual, residual))

        result = minimize_scalar(
            objective,
            bounds=(0.12, 0.45),
            method="bounded",
            options={"xatol": 1e-14},
        )
        self.absorption = float(np.clip(result.x, 0.12, 0.45))
        return self

    def predict(self, experiments):
        """Return predictions in the same order as ``experiments``."""
        return np.asarray(
            [transport(experiment, self.absorption) for experiment in experiments],
            dtype=float,
        )
