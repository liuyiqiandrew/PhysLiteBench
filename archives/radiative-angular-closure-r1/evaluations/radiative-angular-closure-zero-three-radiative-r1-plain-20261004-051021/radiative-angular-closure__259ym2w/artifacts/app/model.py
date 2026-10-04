import numpy as np
from scipy.optimize import minimize_scalar


def transport(experiment, absorption=0.0):
    """Return the weak-response signal for one experiment.

    A photon observed at ``(position, time)`` in beam ``b`` was at
    ``position - direction_b[0] * time`` at preparation time. Consequently
    the spatial cosine is translated by that distance, while absorption
    contributes the common survival factor ``exp(-absorption * time)``.
    """
    directions = np.asarray(experiment["directions"], dtype=float)
    weights = np.asarray(experiment["weights"], dtype=float)
    modulations = np.asarray(experiment["modulations"], dtype=float)
    wavenumber = float(experiment["wavenumber"])
    time = float(experiment["time"])
    position = float(experiment["position"])

    # The derivative with respect to epsilon is linear in each beam's
    # weight and modulation. Only the x component of a direction matters,
    # because the preparation varies only with x.
    phase = wavenumber * (position - directions[:, 0] * time)
    response = np.sum(weights * modulations * np.cos(phase))
    return float(np.exp(-float(absorption) * time) * response)


class Model:
    def __init__(self):
        self.absorption = .25

    def fit(self, records):
        """Fit the common absorption rate to weighted calibration records."""
        records = list(records)
        if not records:
            return self

        # Each experiment has the form response_i * exp(-kappa * time_i),
        # leaving only a one-dimensional optimization for the unknown rate.
        responses = np.asarray(
            [transport(record["input"]) for record in records], dtype=float
        )
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigmas = np.asarray([record["sigma"] for record in records], dtype=float)
        times = np.asarray(
            [record["input"]["time"] for record in records], dtype=float
        )

        def chi_squared(kappa):
            residual = (responses * np.exp(-kappa * times) - values) / sigmas
            return float(np.dot(residual, residual))

        result = minimize_scalar(
            chi_squared,
            bounds=(0.12, 0.45),
            method="bounded",
            options={"xatol": 1e-12},
        )
        self.absorption = float(np.clip(result.x, 0.12, 0.45))
        return self

    def predict(self, experiments):
        """Return predictions in the same order as ``experiments``."""
        return np.asarray(
            [transport(experiment, self.absorption) for experiment in experiments],
            dtype=float,
        )
