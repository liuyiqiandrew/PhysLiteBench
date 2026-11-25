"""Analytic predictions for the pulsed gas-density apparatus."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import jv


_MIN_FRICTION = 0.08
_MAX_FRICTION = 0.4


def _kick_displacement(t, gamma):
    """Mean displacement caused by a unit velocity kick over time ``t``."""
    return -np.expm1(-gamma * t) / gamma


def _thermal_displacement_variance(t, gamma, thermal_speed):
    """Variance of free displacement over an interval of length ``t``."""
    # The stationary OU velocity covariance is
    # thermal_speed**2 * exp(-gamma*abs(t1-t2)).
    return (
        2.0
        * thermal_speed**2
        * (gamma * t + np.expm1(-gamma * t))
        / gamma**2
    )


def response(e, gamma, cutoff=64):
    """Return the requested signed Fourier component for one experiment."""
    gamma = float(gamma)
    tau = float(e["delay"])
    after = float(e["after"])
    thermal_speed = float(e["thermal_speed"])
    a1 = float(e["a1"])
    a2 = float(e["a2"])
    k1 = int(e["k1"])
    k2 = int(e["k2"])
    mode = int(e["mode"])

    a_tau = _kick_displacement(tau, gamma)
    a_after = _kick_displacement(after, gamma)
    a_total = _kick_displacement(tau + after, gamma)

    var_tau = _thermal_displacement_variance(tau, gamma, thermal_speed)
    var_after = _thermal_displacement_variance(after, gamma, thermal_speed)

    # If Z_tau is the free displacement up to the second kick and Z_after is
    # the subsequent free displacement, stationarity gives this covariance.
    cov_tau_after = thermal_speed**2 * a_tau * a_after

    n_all = np.arange(-int(cutoff), int(cutoff) + 1, dtype=int)
    # The initial uniform position imposes m + n*k2 + ell*k1 = 0.
    keep = (mode + n_all * k2) % k1 == 0
    n = n_all[keep]
    q = mode + n * k2
    ell = -q // k1

    # The coefficient of Z_tau is q and the coefficient of Z_after is mode.
    variance = (
        q * q * var_tau
        + mode * mode * var_after
        + 2.0 * q * mode * cov_tau_after
    )

    second_argument = mode * a2 * a_after
    first_argument = a1 * (mode * a_total + n * k2 * a_tau)
    terms = (
        jv(n, second_argument)
        * jv(ell, first_argument)
        * np.exp(
            -0.5 * variance
            + 1j * (n * float(e["phase2"]) + ell * float(e["phase1"]))
        )
    )
    value = np.sum(terms, dtype=np.complex128)
    return float(value.real if e["component"] == "cos" else value.imag)


def predict_at(experiments, friction):
    """Return predictions in the input order as a one-dimensional array."""
    if friction is None:
        raise ValueError("fit must be called before predict")
    return np.asarray(
        [response(experiment, float(friction)) for experiment in experiments],
        dtype=float,
    )


class Model:
    def __init__(self):
        self.friction = None

    def fit(self, records):
        """Fit the common friction by weighted least squares."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record["input"] for record in records]
        observed = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)
        if np.any(~np.isfinite(observed)) or np.any(~np.isfinite(sigma)):
            raise ValueError("calibration values and sigmas must be finite")
        if np.any(sigma <= 0.0):
            raise ValueError("calibration sigmas must be positive")

        def objective(gamma):
            residual = (predict_at(experiments, gamma) - observed) / sigma
            return float(np.dot(residual, residual))

        # A short grid makes the fit robust to small oscillations in the
        # Bessel response; bounded minimization gives sub-grid accuracy.
        grid = np.linspace(_MIN_FRICTION, _MAX_FRICTION, 129)
        grid_objective = np.asarray([objective(gamma) for gamma in grid])
        best = int(np.argmin(grid_objective))
        if best == 0 or best == len(grid) - 1:
            result = minimize_scalar(
                objective,
                bounds=(_MIN_FRICTION, _MAX_FRICTION),
                method="bounded",
                options={"xatol": 1e-13, "maxiter": 300},
            )
        else:
            result = minimize_scalar(
                objective,
                bounds=(grid[best - 1], grid[best + 1]),
                method="bounded",
                options={"xatol": 1e-13, "maxiter": 300},
            )

        candidates = [(float(result.fun), float(result.x))]
        candidates.extend(
            (float(grid_objective[i]), float(grid[i])) for i in (0, -1)
        )
        self.friction = min(candidates, key=lambda item: item[0])[1]
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.friction)


__all__ = ["Model", "predict_at", "response"]
