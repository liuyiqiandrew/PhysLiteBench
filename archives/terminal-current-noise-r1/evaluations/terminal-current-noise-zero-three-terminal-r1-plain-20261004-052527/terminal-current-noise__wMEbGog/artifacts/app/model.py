"""Sequential-tunnelling model for the two-state device."""

import numpy as np
from scipy.special import expit


def transition_rates(e, rate):
    """Return entry and exit rates for the left and right junctions."""
    fraction = float(e["fraction"])
    voltages = np.array(
        [e["voltage_left"], e["voltage_right"]], dtype=float
    )
    epsilon = (
        float(e["offset"])
        - fraction * voltages[0]
        - (1.0 - fraction) * voltages[1]
    )
    bare = float(rate) * np.array(
        [e["left_factor"], e["right_factor"]], dtype=float
    )
    fermi = expit(-(epsilon + voltages) / float(e["thermal_energy"]))
    return bare * fermi, bare * (1.0 - fermi)


def _current_matrices(e, rate):
    """Build the generator and marked-jump matrices for the left wire."""
    entry, exit = transition_rates(e, rate)
    entry_total = float(np.sum(entry))
    exit_total = float(np.sum(exit))
    generator = np.array(
        [[-entry_total, exit_total], [entry_total, -exit_total]], dtype=float
    )
    stationary_prob = np.array([exit_total, entry_total], dtype=float)
    stationary_prob /= entry_total + exit_total

    # A charge change of the island induces displacement current through both
    # capacitors.  Thus the left-wire marks for entry from left/right are
    # 1-C_left/C_sum and -C_left/C_sum; exit marks are their negatives.
    fraction = float(e["fraction"])
    marks = np.array([1.0 - fraction, -fraction], dtype=float)
    current = np.array(
        [[0.0, -float(np.dot(marks, exit))],
         [float(np.dot(marks, entry)), 0.0]],
        dtype=float,
    )
    square = np.array(
        [[0.0, float(np.dot(marks * marks, exit))],
         [float(np.dot(marks * marks, entry)), 0.0]],
        dtype=float,
    )
    return generator, stationary_prob, current, square


def spectrum(e, rate):
    """Return the connected stationary current-noise spectrum."""
    generator, p, current, square = _current_matrices(e, rate)
    stationary = np.outer(p, np.ones(2))
    omega = float(e["omega"])
    resolvent = np.linalg.solve(
        1j * omega * np.eye(2) - generator + stationary,
        np.eye(2) - stationary,
    )
    ones = np.ones(2)
    white = 2.0 * ones @ square @ p
    correlation = 4.0 * np.real(ones @ current @ resolvent @ current @ p)
    return max(0.0, float(white + correlation))


def predict_at(experiments, rate):
    return np.asarray([spectrum(e, rate) for e in experiments])


class Model:
    def __init__(self):
        self.rate = None

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError("fit requires at least one calibration record")
        basis = predict_at((record["input"] for record in records), 1.0)
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)
        if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise ValueError("calibration sigmas must be positive and finite")
        scale = 1.0 / sigma
        numerator = np.dot(basis * scale, values * scale)
        denominator = np.dot(basis * scale, basis * scale)
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("calibration records have no usable signal")
        self.rate = float(np.clip(numerator / denominator, 0.7, 1.4))
        return self

    def predict(self, experiments):
        if self.rate is None:
            raise RuntimeError("fit must be called before predict")
        result = predict_at(experiments, self.rate)
        if not np.isfinite(result).all():
            raise ValueError("non-finite experiment prediction")
        return result
