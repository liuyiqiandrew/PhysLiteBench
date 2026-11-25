import numpy as np
from scipy.optimize import brentq


def response(e):
    """Long-time readout at unit speed, averaged over the length order.

    Removing the ordered rod lengths leaves a Poisson gas of density c.
    For positive current j and mean crossed length a, its current rate
    function is the supremum over s of

        s*j - c*(cosh(s)-1) + c*a*j*sinh(s).

    Averaging over the lengths of the j*t crossed rods replaces the
    length cost c*a*sinh(s) by -log E[exp(-c*ell*sinh(s))].  Consequently
    |field| = s - log E[exp(-c*ell*sinh(s))], and the SCGF is
    c*(cosh(s)-1).  The two-point length distribution gives the log-cosh
    term below.  Replacing ell by its mean would lose this annealed term.
    """
    rho, mean = e["density"], e["length"]
    statistic = e["statistic"]
    if statistic == "mean":
        return 0.0
    if statistic == "variance":
        # Length fluctuations do not enter the quadratic term at zero field.
        return rho * (1.0 - rho * mean)
    if statistic != "scgf":
        raise ValueError(f"Unknown statistic: {statistic!r}")

    field = abs(e["counting_field"])
    if field == 0.0:
        return 0.0
    contrast = e["contrast"]
    c = rho / (1.0 - rho * mean)

    def equation(s):
        excluded = c * mean * np.sinh(s)
        # log(cosh(z)) in a form accurate near zero and safe on this domain.
        log_cosh = np.log1p(2.0 * np.sinh(0.5 * contrast * excluded) ** 2)
        return s + excluded - log_cosh - field

    # The equation is strictly increasing since every possible length is
    # positive.  Its unique root lies between zero and the absolute field.
    s = brentq(equation, 0.0, field, xtol=np.nextafter(0.0, 1.0))
    return c * 2.0 * np.sinh(s / 2.0) ** 2


def predict_at(experiments, speed):
    return speed * np.asarray([response(e) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        """Fit the shared speed by Gaussian weighted least squares."""
        records = list(records)
        if not records:
            raise ValueError("At least one calibration record is required.")
        unit = np.asarray([response(r["input"]) for r in records])
        values = np.asarray([r["value"] for r in records], dtype=float)
        sigma = np.asarray([r["sigma"] for r in records], dtype=float)
        if not np.all(np.isfinite(values)) or not np.all(
            np.isfinite(sigma) & (sigma > 0.0)
        ):
            raise ValueError("Calibration values must be finite and sigma positive.")
        weighted_unit = unit / sigma
        information = np.dot(weighted_unit, weighted_unit)
        if information == 0.0:
            raise ValueError("Calibration records do not identify the speed.")
        estimate = np.dot(weighted_unit, values / sigma) / information
        self.speed = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.speed is None:
            raise ValueError("Fit the model before predicting.")
        return predict_at(experiments, self.speed)
