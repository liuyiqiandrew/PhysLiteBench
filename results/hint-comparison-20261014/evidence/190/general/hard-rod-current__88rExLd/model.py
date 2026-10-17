"""Response functions for the ballistic hard-rod current experiment."""

import numpy as np
from scipy.optimize import brentq


def _log_cosh(x):
    """``log(cosh(x))`` without needlessly overflowing."""
    ax = abs(x)
    return ax + np.log1p(np.exp(-2.0 * ax)) - np.log(2.0)


def response(experiment):
    """Dimensionless response (the speed has deliberately been factored out).

    In reduced coordinates the rods form a gas of density ``c``.  For a
    counting field h, write s for the free-gas counting field.  Annealing the
    two possible rod lengths gives, for h >= 0,

        h = s + c*l*sinh(s) - log(cosh(c*l*contrast*sinh(s))).

    The last term is absent for equal rods.  It is important for the SCGF,
    although it has no quadratic contribution to it: consequently the
    variance is independent of contrast.  The response itself is
    c * (cosh(s) - 1).  The negative-field answer follows by symmetry.
    """
    rho = float(experiment["density"])
    length = float(experiment["length"])
    contrast = float(experiment["contrast"])
    statistic = experiment["statistic"]

    if statistic == "mean":
        return 0.0
    if statistic == "variance":
        return rho * (1.0 - rho * length)
    if statistic != "scgf":
        raise ValueError("statistic must be 'mean', 'variance', or 'scgf'")

    field = abs(float(experiment["counting_field"]))
    if field == 0.0:
        return 0.0

    c = rho / (1.0 - rho * length)
    cl = c * length
    cd = cl * contrast

    def field_at(s):
        sh = np.sinh(s)
        return s + cl * sh - _log_cosh(cd * sh)

    # field_at is strictly increasing in the apparatus domain.  Its value at
    # ``field + 1`` is already above field, but grow the bracket defensively
    # so this remains reliable for benign out-of-domain callers too.
    upper = field + 1.0
    while field_at(upper) < field:
        upper *= 2.0
    s = brentq(lambda x: field_at(x) - field, 0.0, upper, xtol=1e-13)
    return c * (np.cosh(s) - 1.0)


def predict_at(experiments, speed):
    return float(speed) * np.asarray([response(e) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        """Fit the common multiplicative speed by Gaussian weighted LS."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        base = np.asarray([response(record["input"]) for record in records])
        values = np.asarray([float(record["value"]) for record in records])
        sigma = np.asarray([float(record["sigma"]) for record in records])
        if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise ValueError("calibration sigmas must be finite and positive")

        weights = sigma ** -2
        denominator = np.dot(weights, base * base)
        if not np.isfinite(denominator) or denominator == 0.0:
            raise ValueError("calibration records contain no speed information")
        self.speed = float(np.dot(weights, base * values) / denominator)
        return self

    def predict(self, experiments):
        if self.speed is None:
            raise RuntimeError("fit must be called before predict")
        return predict_at(experiments, self.speed)
