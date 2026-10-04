"""Model for the weak radiation-pattern transport experiment.

For a beam with direction ``n``, the transport equation is

    dI/dt + n_x dI/dx = -kappa I.

Consequently, only the x component of a direction enters the measurement:
the initial cosine is translated by ``n_x * time`` and attenuated by the
same factor for every beam.
"""

import numpy as np
from scipy.optimize import minimize_scalar


def transport(experiment):
    """Return the weak-response signal before common absorption."""
    directions = np.asarray(experiment["directions"], dtype=float)
    weights = np.asarray(experiment["weights"], dtype=float)
    modulations = np.asarray(experiment["modulations"], dtype=float)
    wavenumber = int(experiment["wavenumber"])
    time = float(experiment["time"])
    position = float(experiment["position"])

    # The spatial domain is periodic, so the phase can be evaluated directly
    # without any special boundary handling.
    phases = wavenumber * (position - directions[:, 0] * time)
    return float(np.sum(weights * modulations * np.cos(phases)))


class Model:
    def __init__(self):
        self.absorption = .25

    def fit(self, records):
        """Fit the common absorption rate and return this model.

        The calibration observations are independent Gaussians, so the
        maximum-likelihood estimate minimizes the sigma-weighted squared
        residuals.  The one-dimensional bounded optimization also handles
        calibrations whose optimum is at a permitted endpoint.
        """
        records = list(records)
        if not records:
            return self

        inputs = [record["input"] for record in records]
        observed = np.asarray([record["value"] for record in records],
                              dtype=float)
        sigma = np.asarray([record["sigma"] for record in records],
                           dtype=float)
        unattenuated = np.asarray([transport(experiment) for experiment in inputs],
                                   dtype=float)
        times = np.asarray([float(experiment["time"]) for experiment in inputs],
                           dtype=float)

        def objective(kappa):
            residual = (unattenuated * np.exp(-kappa * times) - observed) / sigma
            return float(np.dot(residual, residual))

        result = minimize_scalar(objective, bounds=(0.12, 0.45), method="bounded")
        candidates = [(result.fun, result.x),
                      (objective(0.12), 0.12),
                      (objective(0.45), 0.45)]
        _, best_kappa = min(candidates, key=lambda item: item[0])
        self.absorption = float(best_kappa)
        return self

    def predict(self, experiments):
        """Return one finite weak-response prediction for each experiment."""
        predictions = [transport(experiment) *
                       np.exp(-self.absorption * float(experiment["time"]))
                       for experiment in experiments]
        return np.asarray(predictions, dtype=float)
