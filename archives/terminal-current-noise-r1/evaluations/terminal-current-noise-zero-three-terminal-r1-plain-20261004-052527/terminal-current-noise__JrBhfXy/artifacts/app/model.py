"""Kinetic model for the two-state single-electron device."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit


def transition_rates(e, rate):
    """Return entry and exit rates for the left and right leads."""

    fraction = float(e["fraction"])
    voltage = np.array(
        [e["voltage_left"], e["voltage_right"]], dtype=float
    )
    epsilon = (
        float(e["offset"])
        - fraction * voltage[0]
        - (1.0 - fraction) * voltage[1]
    )
    bare = float(rate) * np.array(
        [e["left_factor"], e["right_factor"]], dtype=float
    )
    # The lead chemical potential is -voltage.
    occupation = expit(
        -(epsilon + voltage) / float(e["thermal_energy"])
    )
    return bare * occupation, bare * (1.0 - occupation)


def _current_weights(e):
    """Electron-flow impulse in the left source wire for each junction."""

    fraction = float(e["fraction"])
    # An island electron increase induces -fraction electron charges in the
    # left source wire.  Add the direct left-junction electron impulse.
    return np.array([1.0 - fraction, -fraction], dtype=float)


def spectrum(e, rate):
    """Return the connected stationary left-source current spectrum."""

    entry, exit = transition_rates(e, rate)
    a = float(np.sum(entry))
    b = float(np.sum(exit))
    total = a + b
    stationary_prob = np.array([b, a], dtype=float) / total
    generator = np.array([[-a, b], [a, -b]], dtype=float)

    # Matrices use destination, source ordering, as does the generator.
    weights = _current_weights(e)
    current = np.array(
        [[0.0, -np.dot(weights, exit)],
         [np.dot(weights, entry), 0.0]],
        dtype=float,
    )
    square = np.array(
        [[0.0, np.dot(weights * weights, exit)],
         [np.dot(weights * weights, entry), 0.0]],
        dtype=float,
    )

    ones = np.ones(2, dtype=float)
    stationary_projector = np.outer(stationary_prob, ones)
    omega = float(e["omega"])
    # Integral_0^inf exp(i omega t) (exp(G t)-P) dt.
    resolvent = np.linalg.solve(
        -1j * omega * np.eye(2) - generator + stationary_projector,
        np.eye(2) - stationary_projector,
    )
    white = 2.0 * np.dot(ones, square @ stationary_prob)
    dynamic = 4.0 * np.real(
        np.dot(ones, current @ resolvent @ current @ stationary_prob)
    )
    result = float(white + dynamic)
    if not np.isfinite(result):
        raise FloatingPointError("non-finite current spectrum")
    return max(0.0, result)


def predict_at(experiments, rate):
    """Evaluate ``spectrum`` for each experiment as a 1-D NumPy array."""

    return np.asarray([spectrum(e, rate) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.rate = None

    def fit(self, records):
        """Infer the common bare rate from weighted calibration records."""

        records = list(records)
        if not records:
            self.rate = 1.05
            return self

        experiments = [record["input"] for record in records]
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigmas = np.asarray([record["sigma"] for record in records], dtype=float)
        if (not np.all(np.isfinite(values)) or
                not np.all(np.isfinite(sigmas)) or
                np.any(sigmas <= 0.0)):
            raise ValueError("calibration values and sigmas must be finite and positive")

        def objective(rate):
            residual = (predict_at(experiments, rate) - values) / sigmas
            return float(np.dot(residual, residual))

        lower, upper = 0.7, 1.4
        result = minimize_scalar(
            objective,
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 1e-12},
        )
        candidates = [
            (float(result.fun), float(result.x)),
            (objective(lower), lower),
            (objective(upper), upper),
        ]
        _, best_rate = min(candidates, key=lambda item: item[0])
        self.rate = float(np.clip(best_rate, lower, upper))
        return self

    def predict(self, experiments):
        if self.rate is None:
            raise RuntimeError("fit must be called before predict")
        return predict_at(experiments, self.rate)
