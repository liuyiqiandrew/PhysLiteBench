"""Current SCGF model and calibration interface.

The apparatus is a periodic half-filled symmetric exclusion process on L sites
with weak asymmetry fixed by the dimensionless field E: particles hop right at
rate D*exp(E/L) and left at rate D*exp(-E/L).  Because the local bias is O(1/L),
the large-ring, long-time scaled cumulant generating function of the bond
current Q_t = (J_+ - J_-)/L is Gaussian in the counting parameter lambda:

    psi(E, lambda; D) = D * lambda * (lambda + 2 E) / 4.

The linear-in-lambda term is the mean current (D*E/2 per bond, times L bonds
divided by L) and the quadratic term is the current variance coefficient
sigma(rho)/2 = D*rho*(1-rho) = D/4 at half filling.

Only D is unknown; it is shared by every experiment and estimated from the
calibration records by inverse-variance weighted least squares.

"""
import numpy as np


def predict_at(experiments, diffusivity):
    """Evaluate the asymptotic SCGF psi(E, lambda; D) for a batch of inputs."""
    field = np.asarray([e["field"] for e in experiments], dtype=float)
    bias = np.asarray([e["bias"] for e in experiments], dtype=float)
    return diffusivity * bias * (bias + 2.0 * field) / 4.0


class Model:
    def __init__(self):
        self.diffusivity = 1.0

    def fit(self, records):
        """Estimate the shared diffusivity D from noisy SCGF measurements.

        Each record provides ``input`` (field E, bias lambda), a noisy ``value``
        of psi and the known Gaussian error ``sigma``.  The model is linear in
        the single parameter D, so the weighted least-squares estimate is a
        closed-form ratio; it is clipped to the documented physical range.
        """
        records = list(records)
        if not records:
            return self

        field = np.asarray([r["input"]["field"] for r in records], dtype=float)
        bias = np.asarray([r["input"]["bias"] for r in records], dtype=float)
        value = np.asarray([r["value"] for r in records], dtype=float)
        sigma = np.asarray([r["sigma"] for r in records], dtype=float)

        # Regression coefficient g such that psi = D * g.
        g = bias * (bias + 2.0 * field) / 4.0

        weight = 1.0 / sigma ** 2
        denom = float(np.sum(weight * g * g))
        if denom > 0.0:
            diffusivity = float(np.sum(weight * g * value) / denom)
        else:
            # No informative records: fall back to the weighted mean level.
            diffusivity = float(np.sum(weight * value) / np.sum(weight))

        self.diffusivity = float(np.clip(diffusivity, 0.8, 1.2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
