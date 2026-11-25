"""Model for the elastic slab with a continuum of local resonators.

The fields are harmonic with the convention ``exp(-i * frequency * t)``.
Inside the slab the resonators can be eliminated from the equation of motion,
but their kinetic and spring energies must still be included in the reported
energy.  The latter point is important: the effective density used to find
the wave number is not, by itself, the energy density of this dispersive
medium.
"""

import numpy as np


LEAD_IMPEDANCE = np.sqrt(2.0)
HOST_STIFFNESS_BOUNDS = (0.7, 1.6)


def _scattering_data(experiment, host_stiffness):
    """Return the slab wave and the integrated squared field quantities.

    The displacement in the slab is represented as

        u(x) = a cos(q x) + b sin(q x).

    The returned integrals are ``int |u|^2 dx`` and ``int |u'|^2 dx`` over
    the slab.  ``alpha`` is the resonator displacement divided by the host
    displacement.
    """
    length = float(experiment["length"])
    frequency = float(experiment["frequency"])
    mass = float(experiment["resonator_mass"])
    resonance = float(experiment["resonance_frequency"])
    amplitude = float(experiment["incident_amplitude"])

    # v = alpha*u follows from the local resonator equation.  With no
    # resonator, avoid evaluating an irrelevant possible 0/0 expression.
    if mass == 0.0:
        alpha = 1.0
        density = 1.0
    else:
        denominator = resonance * resonance - frequency * frequency
        alpha = resonance * resonance / denominator
        density = 1.0 + mass * resonance * resonance / denominator

    wave_number = frequency * np.sqrt(density / host_stiffness)
    cosine = np.cos(wave_number * length)
    sine = np.sin(wave_number * length)
    lead_impedance = 1j * frequency * LEAD_IMPEDANCE
    slab_impedance = host_stiffness * wave_number

    # At x=0, traction + i*Z*u = 2*i*Z*incident_amplitude.  At x=L,
    # traction - i*Z*u = 0 (there is no incoming wave from the right).
    matrix = np.array(
        [
            [lead_impedance, slab_impedance],
            [
                -slab_impedance * sine - lead_impedance * cosine,
                slab_impedance * cosine - lead_impedance * sine,
            ],
        ],
        dtype=complex,
    )
    a, b = np.linalg.solve(matrix, [2.0 * lead_impedance * amplitude, 0.0])

    # Integrals of cos^2, sin^2, and sin*cos over [0, length].
    cosine_squared = length / 2.0 + np.sin(2.0 * wave_number * length) / (
        4.0 * wave_number
    )
    sine_squared = length - cosine_squared
    cross = np.sin(wave_number * length) ** 2 / (2.0 * wave_number)
    cross_term = 2.0 * np.real(np.conjugate(a) * b) * cross

    displacement_integral = (
        abs(a) ** 2 * cosine_squared
        + abs(b) ** 2 * sine_squared
        + cross_term
    )
    strain_integral = wave_number * wave_number * (
        abs(a) ** 2 * sine_squared
        + abs(b) ** 2 * cosine_squared
        - cross_term
    )

    return density, alpha, displacement_integral, strain_integral


def wave_data(experiment, host_stiffness):
    """Compatibility helper returning effective density and host integrals."""
    density, _alpha, displacement, strain = _scattering_data(
        experiment, host_stiffness
    )
    return density, displacement, strain


def _energy(experiment, host_stiffness):
    _density, alpha, displacement, strain = _scattering_data(
        experiment, host_stiffness
    )

    frequency = float(experiment["frequency"])
    mass = float(experiment["resonator_mass"])
    resonance = float(experiment["resonance_frequency"])

    # For a complex harmonic amplitude, the cycle-averaged kinetic and
    # potential energies carry a factor 1/4.  Every resonator has
    # displacement alpha*u and spring extension (alpha-1)*u.
    resonator_energy_density = mass * (
        frequency * frequency * alpha * alpha
        + resonance * resonance * (alpha - 1.0) ** 2
    )
    return (
        frequency * frequency * displacement
        + resonator_energy_density * displacement
        + host_stiffness * strain
    ) / 4.0


def predict_at(experiments, host_stiffness):
    """Predict cycle-averaged slab energies in input order."""
    return np.asarray(
        [_energy(experiment, host_stiffness) for experiment in experiments],
        dtype=float,
    )


def _fit_objective(records, host_stiffness):
    experiments = [record["input"] for record in records]
    observed = np.asarray([record["value"] for record in records], dtype=float)
    sigma = np.asarray([record["sigma"] for record in records], dtype=float)
    predicted = predict_at(experiments, host_stiffness)
    residual = (predicted - observed) / sigma
    return float(residual @ residual)


def _bounded_minimize(function, lower, upper):
    """Minimize a scalar function on a short bounded interval.

    A grid makes this robust to the modest interference oscillations in a
    slab, and golden-section refinement gives more than enough precision for
    the one fitted parameter.
    """
    grid = np.linspace(lower, upper, 2001)
    values = np.asarray([function(value) for value in grid])

    # Refine every grid-local minimum.  This avoids assuming that a nonlinear
    # scattering objective is globally unimodal.
    candidates = [0, len(grid) - 1]
    candidates.extend(
        index
        for index in range(1, len(grid) - 1)
        if values[index] <= values[index - 1] and values[index] <= values[index + 1]
    )

    best_x = float(grid[int(np.nanargmin(values))])
    best_value = float(np.nanmin(values))
    golden_ratio = (np.sqrt(5.0) - 1.0) / 2.0

    for index in candidates:
        left = float(grid[max(0, index - 1)])
        right = float(grid[min(len(grid) - 1, index + 1)])
        if right <= left:
            continue
        x1 = right - golden_ratio * (right - left)
        x2 = left + golden_ratio * (right - left)
        f1 = function(x1)
        f2 = function(x2)
        for _ in range(80):
            if f1 <= f2:
                right, x2, f2 = x2, x1, f1
                x1 = right - golden_ratio * (right - left)
                f1 = function(x1)
            else:
                left, x1, f1 = x1, x2, f2
                x2 = left + golden_ratio * (right - left)
                f2 = function(x2)
        x = (left + right) / 2.0
        value = function(x)
        if value < best_value:
            best_x, best_value = x, value

    return best_x


class Model:
    def __init__(self):
        self.host_stiffness = None

    def fit(self, records):
        """Fit the common host stiffness by weighted least squares."""
        records = list(records)
        if not records:
            raise ValueError("At least one calibration record is required.")
        if any(float(record["sigma"]) <= 0.0 for record in records):
            raise ValueError("Calibration standard deviations must be positive.")

        self.host_stiffness = _bounded_minimize(
            lambda stiffness: _fit_objective(records, stiffness),
            *HOST_STIFFNESS_BOUNDS,
        )
        return self

    def predict(self, experiments):
        if self.host_stiffness is None:
            raise ValueError("Model must be fitted before prediction.")
        return predict_at(experiments, self.host_stiffness)
