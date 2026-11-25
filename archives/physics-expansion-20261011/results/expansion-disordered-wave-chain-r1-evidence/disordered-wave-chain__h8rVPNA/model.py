"""Transmission model for a chain of identical delta-function barriers."""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar


# The phase average converges rapidly with this fixed quadrature.  Keeping it
# at module scope also makes predictions inexpensive for repeated experiments.
_phase_nodes, _phase_weights = leggauss(64)
_phase_cos = np.cos(np.pi * (_phase_nodes + 1.0))
_phase_weights = _phase_weights / 2.0  # average over theta in [0, 2*pi]


def _random_chain_transmission(count, q):
    """Return E[T] for ``count`` barriers with dimensionless strength q.

    With R=(1-T)/T=sinh(eta)^2 and sinh(alpha)=q, adding one barrier with
    random phase obeys

      cosh(2 eta') = cosh(2 eta) cosh(2 alpha)
                     + sinh(2 eta) sinh(2 alpha) cos(theta).

    A backward dynamic-programming phase average evaluates this recurrence
    without sampling a multidimensional set of separations.
    """
    q = float(q)
    if count == 0:
        return 1.0
    if count == 1:
        return 1.0 / (1.0 + q * q)
    if count == 2:
        return 1.0 / (1.0 + 2.0 * q * q)

    alpha = np.arcsinh(q)
    grid_size = 400
    eta = np.linspace(0.0, count * alpha, grid_size + 1)
    value = 1.0 / np.cosh(eta) ** 2  # final payoff T=1/(1+R)

    for stage in range(count - 1, -1, -1):
        next_eta = eta
        next_value = value
        eta = np.linspace(0.0, stage * alpha, grid_size + 1)

        sinh_eta = np.sinh(eta)[:, None]
        cosh_eta = np.cosh(eta)[:, None]
        resistance = (
            sinh_eta * sinh_eta
            + q * q
            + 2.0 * q * q * sinh_eta * sinh_eta
            + 2.0 * q * np.sqrt(1.0 + q * q)
            * sinh_eta * cosh_eta * _phase_cos[None, :]
        )
        following_eta = np.arcsinh(np.sqrt(np.maximum(resistance, 0.0)))
        following_eta = np.clip(following_eta, 0.0, next_eta[-1])

        spline = CubicSpline(next_eta, next_value, bc_type="natural")
        value = np.sum(
            spline(following_eta) * _phase_weights[None, :], axis=1
        )

    return float(value[0])


def response(e, strength):
    """Expected power transmission for one experiment."""
    count = int(e["count"])
    wave_number = float(e["wave_number"])
    if count < 0 or wave_number <= 0.0:
        raise ValueError("count must be non-negative and wave_number positive")
    return _random_chain_transmission(count, float(strength) / wave_number)


def predict_at(experiments, strength):
    """Predict a batch, retaining input order and avoiding duplicate work."""
    cache = {}
    predictions = []
    for experiment in experiments:
        key = (int(experiment["count"]), float(experiment["wave_number"]))
        if key not in cache:
            cache[key] = response(experiment, strength)
        predictions.append(cache[key])
    return np.asarray(predictions, dtype=float)


class Model:
    def __init__(self):
        self.strength = None

    def fit(self, records):
        """Fit the common positive barrier strength by weighted least squares."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        def objective(strength):
            total = 0.0
            for record in records:
                sigma = float(record["sigma"])
                if sigma <= 0.0:
                    raise ValueError("calibration sigma must be positive")
                residual = (
                    response(record["input"], strength) - float(record["value"])
                ) / sigma
                total += residual * residual
            return total

        result = minimize_scalar(
            objective,
            bounds=(0.6, 1.0),
            method="bounded",
            options={"xatol": 1e-12},
        )
        if not result.success or not np.isfinite(result.x):
            raise RuntimeError("unable to fit barrier strength")
        self.strength = float(result.x)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.strength)
