"""Surface-wave model for the loaded compressible neo-Hookean half-space."""

import numpy as np
from scipy.optimize import brentq


MU = 1.0e6
LAMBDA_OVER_MU = 2.0
MIN_DENSITY = 900.0
MAX_DENSITY = 1300.0


def geometry(e):
    """Return the imposed tangential and equilibrium normal stretches."""

    s = float(e["stretch"])
    pressure = float(e["pressure"]) / MU

    # The normal Cauchy stress is -pressure.  With J=s*t this gives
    # t^2 - 1 + 2 log(J) + pressure*J = 0.
    def balance(t):
        return t * t - 1.0 + LAMBDA_OVER_MU * np.log(s * t) + pressure * s * t

    t = brentq(balance, 0.2, 2.0, xtol=1.0e-14, rtol=1.0e-14)
    return s, t


def surface_matrix(x, s, t):
    """Return the two-traction secular matrix at dimensionless speed ``x``.

    ``x`` is the shear-wave parameter used to write the phase-speed factor as
    s**2 - t**2 + t**2*x.  Static force balance gives the dimensionless
    pressure-area product as ``c - t**2``.
    """

    b = t * t
    jacobian = s * t
    c = 1.0 - LAMBDA_OVER_MU * np.log(jacobian)
    pressure_area = c - b

    longitudinal = np.sqrt(1.0 - b * x / (b + 2.0 + c))
    transverse = np.sqrt(1.0 - x)

    # The columns are the longitudinal and shear partial waves.  They are
    # scaled to avoid the 1/transverse factor in the shear eigenvector.
    return np.array(
        [
            [
                longitudinal * (b + c - pressure_area),
                b * transverse * transverse + c - pressure_area,
            ],
            [
                2.0 - (b + c + 2.0) * longitudinal * longitudinal
                + pressure_area,
                -(b + c - pressure_area) * transverse,
            ],
        ],
        dtype=float,
    )


def mode(e):
    """Return rho * (phase speed)**2 / MU for one experiment."""

    s, t = geometry(e)

    # The secular determinant has a trivial zero at x=0.  The requested
    # branch is the other subsonic zero in (0, 1).
    def secular(x):
        return np.linalg.det(surface_matrix(x, s, t)) / x

    x = brentq(secular, 1.0e-8, 1.0 - 1.0e-10, xtol=1.0e-13, rtol=1.0e-14)
    return s * s - t * t + t * t * x


def predict_at(experiments, density):
    """Predict phase speeds for a known undeformed density."""

    if density is None or not np.isfinite(density) or density <= 0.0:
        raise RuntimeError("fit must be called before predict")
    dimensionless = np.asarray([mode(e) for e in experiments], dtype=float)
    return np.sqrt(MU * dimensionless / density)


class Model:
    def __init__(self):
        self.density = None

    def fit(self, records):
        """Fit the common undeformed density from calibration records."""

        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        # For fixed experiments, speed = sqrt(MU*mode) / sqrt(density).
        # Thus z=1/sqrt(density) is a linear parameter and the Gaussian
        # maximum-likelihood estimate is a one-parameter weighted fit.
        scale = np.asarray([np.sqrt(MU * mode(r["input"])) for r in records])
        values = np.asarray([float(r["value"]) for r in records])
        sigma = np.asarray([float(r["sigma"]) for r in records])
        if (
            not np.isfinite(scale).all()
            or not np.isfinite(values).all()
            or not np.isfinite(sigma).all()
            or np.any(sigma <= 0.0)
        ):
            raise ValueError("calibration values and uncertainties must be finite")

        weights = 1.0 / sigma**2
        z = np.sum(weights * scale * values) / np.sum(weights * scale**2)
        if z <= 0.0 or not np.isfinite(z):
            raise ValueError("calibration records do not determine a positive density")

        density = 1.0 / (z * z)
        self.density = float(np.clip(density, MIN_DENSITY, MAX_DENSITY))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.density)
