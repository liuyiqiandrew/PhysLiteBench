"""Current SCGF model and calibration interface."""
import numpy as np

# Physical bounds for the unknown diffusivity.
_D_MIN, _D_MAX = 0.8, 1.2


def predict_at(experiments, diffusivity):
    """Leading large-ring SCGF psi(E, lambda; D) for the ring SEP.

    The tilted generator for the symmetric exclusion process on a ring with a
    bias contributes an additive field to the counting parameter.  Its dominant
    eigenvalue scales as D * lambda * (lambda + 2 E) / 4 at leading order in
    1/L, which is the observable reported here.
    """
    field = np.asarray([e["field"] for e in experiments], dtype=float)
    bias = np.asarray([e["bias"] for e in experiments], dtype=float)
    return diffusivity * bias * (bias + 2.0 * field) / 4.0


def _basis(experiments):
    """Design column v(E, lambda) such that psi = D * v(E, lambda)."""
    field = np.asarray([e["field"] for e in experiments], dtype=float)
    bias = np.asarray([e["bias"] for e in experiments], dtype=float)
    return bias * (bias + 2.0 * field) / 4.0


class Model:
    def __init__(self):
        self.diffusivity = 1.0

    def fit(self, records):
        """Estimate the diffusivity from noisy SCGF observations.

        The observable is linear in D, so the maximum-likelihood estimate under
        independent Gaussian errors is a weighted least-squares average of the
        per-record diffusivity estimates, using the known sigmas as weights.
        """
        basis = _basis([r["input"] for r in records])
        values = np.asarray([r["value"] for r in records], dtype=float)
        sigma = np.asarray([r["sigma"] for r in records], dtype=float)

        weight = 1.0 / sigma ** 2
        denominator = float(np.sum(weight * basis ** 2))
        if denominator > 0.0:
            estimate = float(np.sum(weight * basis * values) / denominator)
        else:
            estimate = 1.0

        if not np.isfinite(estimate):
            estimate = 1.0
        self.diffusivity = float(np.clip(estimate, _D_MIN, _D_MAX))
        return self

    def predict(self, experiments):
        experiments = list(experiments)
        if not experiments:
            return np.empty(0, dtype=float)
        return predict_at(experiments, self.diffusivity)
