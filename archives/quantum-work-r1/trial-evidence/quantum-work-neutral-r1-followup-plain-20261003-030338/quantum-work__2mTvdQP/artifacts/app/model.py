"""Prediction model for the driven three-level apparatus.

The energy record is a two-point measurement record.  Consequently, its
probability distribution is obtained from the initial Gibbs populations and
the transition probabilities between the initial and final energy bases; it
is not obtained by treating ``U.H @ H @ U - H`` as an observable.
"""

from functools import lru_cache

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar


D = np.diag([0.0, 1.0, 2.35])
X = np.array([[0.0, 1.0, 0.25], [1.0, 0.0, 0.8], [0.25, 0.8, 0.0]])
Y = np.array(
    [[0.0, -1.0j, 0.45j], [1.0j, 0.0, -0.7j], [-0.45j, 0.7j, 0.0j]]
)

_INPUT_KEYS = (
    "temperature",
    "amplitude_a",
    "amplitude_b",
    "phase",
    "time_a",
    "time_b",
)
_SCALE_BOUNDS = (0.8, 1.2)


@lru_cache(maxsize=4096)
def evolution(scale, amplitude_a, amplitude_b, phase, time_a, time_b):
    """Return the total unitary, with the second pulse applied last."""

    h0 = scale * D
    h_a = h0 + amplitude_a * X
    h_b = h0 + amplitude_b * (np.cos(phase) * X + np.sin(phase) * Y)
    return expm(-1j * time_b * h_b) @ expm(-1j * time_a * h_a)


def state_and_evolution(
    temperature, amplitude_a, amplitude_b, phase, time_a, time_b, scale
):
    """Return the H0 energies, Gibbs populations, and driven unitary."""

    energies = scale * np.diag(D)
    # Subtracting the minimum energy is harmless and keeps this general.
    boltzmann = np.exp(-(energies - energies.min()) / temperature)
    populations = boltzmann / boltzmann.sum()
    unitary = evolution(
        float(scale),
        float(amplitude_a),
        float(amplitude_b),
        float(phase),
        float(time_a),
        float(time_b),
    )
    return energies, populations, unitary


def statistics(
    temperature, amplitude_a, amplitude_b, phase, time_a, time_b, scale
):
    """Return mean, variance, and third central moment of the energy record.

    If the initial energy is E_n and the final energy is E_m, the record is
    E_m - E_n with probability p_n * |U[m, n]|**2.  The returned quantities
    are the ordinary population cumulants requested by the interface.
    """

    energies, populations, unitary = state_and_evolution(
        temperature, amplitude_a, amplitude_b, phase, time_a, time_b, scale
    )

    # Rows correspond to the initially occupied state n; columns to final m.
    transition_probs = np.abs(unitary) ** 2
    transition_probs = transition_probs.T
    energy_records = energies[None, :] - energies[:, None]
    joint_weights = populations[:, None] * transition_probs

    raw = np.array(
        [np.sum(joint_weights * energy_records**power) for power in (1, 2, 3)],
        dtype=float,
    )
    mean = raw[0]
    variance = raw[1] - mean**2
    third_central = raw[2] - 3.0 * mean * raw[1] + 2.0 * mean**3
    return np.array([mean, variance, third_central], dtype=float)


def predict_at(experiments, scale):
    """Predict the requested cumulant for each experiment in input order."""

    scale = float(scale)
    if not np.isfinite(scale):
        raise ValueError("energy_scale must be finite; call fit first")

    # Several calibration records share an identical pulse sequence at
    # different temperatures/cumulants, so avoid recomputing those statistics.
    cache = {}
    values = []
    for experiment in experiments:
        key = tuple(float(experiment[name]) for name in _INPUT_KEYS)
        if key not in cache:
            cache[key] = statistics(*key, scale)
        cumulant = int(experiment["cumulant"])
        if cumulant not in (1, 2, 3):
            raise ValueError("cumulant must be one of 1, 2, or 3")
        values.append(cache[key][cumulant - 1])
    return np.asarray(values, dtype=float)


class Model:
    def __init__(self):
        self.energy_scale = None

    def fit(self, records):
        """Fit the common energy scale by weighted least squares."""

        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record["input"] for record in records]
        observed = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)
        if not np.isfinite(observed).all() or not np.isfinite(sigma).all():
            raise ValueError("calibration values and uncertainties must be finite")
        if np.any(sigma <= 0.0):
            raise ValueError("calibration uncertainties must be positive")

        def objective(scale):
            residual = (predict_at(experiments, scale) - observed) / sigma
            return float(residual @ residual)

        result = minimize_scalar(
            objective,
            bounds=_SCALE_BOUNDS,
            method="bounded",
            options={"xatol": 1e-12, "maxiter": 200},
        )
        # Include endpoints explicitly in case the optimum is on a bound.
        candidates = [(float(result.fun), float(result.x))]
        candidates.extend((objective(bound), bound) for bound in _SCALE_BOUNDS)
        _, best_scale = min(candidates, key=lambda item: item[0])
        self.energy_scale = float(np.clip(best_scale, *_SCALE_BOUNDS))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.energy_scale)
