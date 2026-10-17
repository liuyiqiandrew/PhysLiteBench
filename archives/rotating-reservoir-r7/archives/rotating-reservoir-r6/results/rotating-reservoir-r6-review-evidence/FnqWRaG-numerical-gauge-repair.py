"""Stationary model for the collisional rotor.

The velocity of the rotor is a continuous state.  Collisions are therefore
not well represented by replacing their rate by its average: the rate depends
on the current velocity and the collision mark is correlated with that rate.
This module solves the stationary forward equation on a velocity grid and
uses the corresponding Poisson equation for the noise readout.
"""

from functools import lru_cache
from math import erf, pi, sqrt

import numpy as np


RATE = 3.6
INPUT_FIELDS = (
    "angular_speed",
    "flip_rate",
    "temperature_a",
    "temperature_b",
    "flux_fraction",
)

GRID_SIZE = 241
GRID_LIMIT = 8.0


def _normal_density(x, mean, variance):
    return np.exp(-0.5 * ((x - mean) ** 2) / variance) / sqrt(2.0 * pi * variance)


def _ou_forward_matrix(x, spacing, drag, temperature):
    """Finite-volume forward generator for the laboratory OU process."""
    n = len(x)
    matrix = np.zeros((n, n), dtype=float)
    diffusion = drag * temperature
    for i in range(n - 1):
        interface = 0.5 * (x[i] + x[i + 1])
        drift = -drag * interface
        left = (0.5 * drift + diffusion / spacing) / spacing
        right = (0.5 * drift - diffusion / spacing) / spacing
        matrix[i, i] -= left
        matrix[i, i + 1] -= right
        matrix[i + 1, i] += left
        matrix[i + 1, i + 1] += right
    return matrix


def _collision_data(x, temperature_b, u, flux_fraction):
    """Return discrete collision intensities and heat marks."""
    spacing = x[1] - x[0]
    gas_density = _normal_density(x, u, temperature_b)
    constant_channel = gas_density[None, :] * spacing
    flux_channel = np.abs(x[:, None] - x[None, :]) * constant_channel
    intensities = RATE * (
        (1.0 - flux_fraction) * constant_channel
        + flux_fraction * flux_channel
    )
    heat = 0.5 * ((x[:, None] - u) ** 2 - (x[None, :] - u) ** 2)
    return intensities, heat


def _generator(experiment, drag, grid_size=GRID_SIZE):
    """Build the forward generator and collision reward arrays."""
    angular_speed = float(experiment["angular_speed"])
    flip_rate = float(experiment["flip_rate"])
    temperature_a = float(experiment["temperature_a"])
    temperature_b = float(experiment["temperature_b"])
    flux_fraction = float(experiment["flux_fraction"])

    spacing = 2.0 * GRID_LIMIT / grid_size
    x = -GRID_LIMIT + (np.arange(grid_size) + 0.5) * spacing
    size = 2 * grid_size
    forward = np.zeros((size, size), dtype=float)
    weights = []
    heat = []

    for state, sign in enumerate((1.0, -1.0)):
        block = slice(state * grid_size, (state + 1) * grid_size)
        forward[block, block] += _ou_forward_matrix(
            x, spacing, drag, temperature_a
        )

        other = 1 - state
        forward[other * grid_size : (other + 1) * grid_size, block] += (
            flip_rate * np.eye(grid_size)
        )
        forward[block, block] -= flip_rate * np.eye(grid_size)

        u = sign * angular_speed
        state_weights, state_heat = _collision_data(
            x, temperature_b, u, flux_fraction
        )
        weights.append(state_weights)
        heat.append(state_heat)

        # state_weights is indexed (pre-collision, post-collision), while a
        # forward generator is indexed (post, pre).
        forward[block, block] += state_weights.T
        forward[block, block] -= np.diag(state_weights.sum(axis=1))

    return x, forward, weights, heat


def _stationary_readouts(experiment, drag, grid_size=GRID_SIZE):
    x, forward, weights, heat = _generator(experiment, float(drag), grid_size)
    n = len(x)

    stationary_system = forward.copy()
    stationary_system[0, :] = 1.0
    stationary_rhs = np.zeros(2 * n)
    stationary_rhs[0] = 1.0
    stationary = np.linalg.solve(stationary_system, stationary_rhs)

    event_mean = np.empty(2 * n)
    event_second = np.empty(2 * n)
    for state in range(2):
        block = slice(state * n, (state + 1) * n)
        event_mean[block] = np.sum(weights[state] * heat[state], axis=1)
        event_second[block] = np.sum(
            weights[state] * heat[state] ** 2, axis=1
        )
    mean = float(stationary @ event_mean)

    # L phi = mean - event_mean.  This Poisson corrector gives the integrated
    # correlation of a collision mark with all subsequent collision marks.
    poisson_system = forward.T.copy()
    pivot = int(np.argmax(stationary))
    poisson_system[pivot, :] = stationary
    poisson_rhs = mean - event_mean
    poisson_rhs[pivot] = 0.0
    corrector = np.linalg.solve(poisson_system, poisson_rhs)

    cross = 0.0
    for state in range(2):
        block = slice(state * n, (state + 1) * n)
        cross += np.sum(
            stationary[block, None]
            * weights[state]
            * heat[state]
            * corrector[block][None, :]
        )
    noise = float(stationary @ event_second + 2.0 * cross)
    return {"mean": mean, "noise": noise}


@lru_cache(maxsize=4096)
def readout(
    angular_speed,
    flip_rate,
    temperature_a,
    temperature_b,
    flux_fraction,
    statistic,
    drag,
):
    if statistic not in ("mean", "noise"):
        raise ValueError("statistic must be mean or noise")
    experiment = dict(
        zip(
            INPUT_FIELDS,
            (
                angular_speed,
                flip_rate,
                temperature_a,
                temperature_b,
                flux_fraction,
            ),
        )
    )
    return _stationary_readouts(experiment, float(drag))[statistic]


def predict_at(experiments, drag):
    return np.asarray(
        [
            readout(
                *(experiment[field] for field in INPUT_FIELDS),
                experiment.get("statistic", "mean"),
                float(drag),
            )
            for experiment in experiments
        ],
        dtype=float,
    )


class Model:
    def __init__(self):
        self.drag = None

    def fit(self, records):
        """Fit the common laboratory drag to the supplied calibration."""
        if not records:
            raise ValueError("records must not be empty")

        # Calibration is at zero flow and zero flux fraction, so the motor is
        # irrelevant.  Collapse repeated noisy measurements first.
        groups = {}
        for record in records:
            experiment = record["input"]
            key = (
                float(experiment["temperature_a"]),
                float(experiment["temperature_b"]),
            )
            sigma = float(record.get("sigma", 1.0))
            if sigma <= 0.0:
                raise ValueError("record sigma must be positive")
            groups.setdefault(key, []).append((float(record["value"]), sigma))

        targets = []
        for (temperature_a, temperature_b), values in groups.items():
            observation = np.array([value for value, _ in values])
            weight = np.array([1.0 / sigma**2 for _, sigma in values])
            targets.append(
                (
                    temperature_a,
                    temperature_b,
                    float(weight @ observation / weight.sum()),
                )
            )

        def objective(drag):
            error = 0.0
            for temperature_a, temperature_b, target in targets:
                experiment = {
                    "angular_speed": 0.0,
                    "flip_rate": 1.0,
                    "temperature_a": temperature_a,
                    "temperature_b": temperature_b,
                    "flux_fraction": 0.0,
                }
                prediction = _stationary_readouts(
                    experiment, drag, grid_size=GRID_SIZE
                )["mean"]
                error += (prediction - target) ** 2
            return error

        # Bounded ternary refinement avoids making scipy a model dependency.
        grid = np.linspace(0.4, 1.1, 15)
        values = np.array([objective(value) for value in grid])
        best = int(np.argmin(values))
        left = grid[max(0, best - 1)]
        right = grid[min(len(grid) - 1, best + 1)]
        if right > left:
            for _ in range(12):
                first = left + (right - left) / 3.0
                second = right - (right - left) / 3.0
                if objective(first) <= objective(second):
                    right = second
                else:
                    left = first
            self.drag = float((left + right) / 2.0)
        else:
            self.drag = float(grid[best])
        return self

    def predict(self, experiments):
        if self.drag is None:
            raise RuntimeError("fit must be called before predict")
        return predict_at(experiments, self.drag)
