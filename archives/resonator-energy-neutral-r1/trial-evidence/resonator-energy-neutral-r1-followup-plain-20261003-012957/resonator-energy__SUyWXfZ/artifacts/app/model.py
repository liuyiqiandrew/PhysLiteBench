"""Scattering model for the resonant elastic slab.

The displacement fields are represented by complex phasors with the time
dependence specified in the README, ``Re(U exp(-i*frequency*t))``.  The
resonators are continuously distributed along the slab, so in a harmonic
state they can be eliminated from the host equation and subsequently added
back when evaluating the energy.
"""

import numpy as np


LEAD_IMPEDANCE = np.sqrt(2.0)
HOST_STIFFNESS_BOUNDS = (0.7, 1.6)


def _wave_integrals(a, b, wave_number, length):
    """Return ``integral |U|^2`` and ``integral |U'|^2`` for the slab field.

    The host displacement phasor is ``U(x) = a*cos(k*x) + b*sin(k*x)``.
    Keeping these integrals analytic avoids quadrature error in the very small
    calibration energies.
    """
    if wave_number == 0.0:
        # This is outside the documented input range, but makes the helper
        # well behaved for a degenerate caller.
        displacement = length * abs(a) ** 2
        strain = length * abs(b * wave_number) ** 2
        return displacement, strain

    sine = np.sin(wave_number * length)
    double_sine = np.sin(2.0 * wave_number * length)
    first = length / 2.0 + double_sine / (4.0 * wave_number)
    second = length - first
    mixed = np.real(np.conjugate(a) * b) * sine**2 / wave_number

    displacement = abs(a) ** 2 * first + abs(b) ** 2 * second + mixed
    strain = wave_number**2 * (
        abs(a) ** 2 * second + abs(b) ** 2 * first - mixed
    )
    return displacement, strain


def wave_data(experiment, host_stiffness):
    """Solve the slab scattering problem and return energy ingredients.

    It returns effective density, integral of ``|U|^2``, and integral of
    ``|U'|^2``.  The resonator displacement ratio is reconstructed in
    ``predict_at`` because resonator and spring energy are not represented by
    the effective density alone.
    """
    length = float(experiment["length"])
    frequency = float(experiment["frequency"])
    mass = float(experiment["resonator_mass"])
    resonance = float(experiment["resonance_frequency"])
    amplitude = float(experiment["incident_amplitude"])
    stiffness = float(host_stiffness)

    omega_squared = frequency**2
    resonance_squared = resonance**2

    if mass:
        denominator = resonance_squared - omega_squared
        density = 1.0 + mass * resonance_squared / denominator
    else:
        density = 1.0

    wave_number = frequency * np.sqrt(density / stiffness)
    cosine = np.cos(wave_number * length)
    sine = np.sin(wave_number * length)

    # In each lead, traction is i*LEAD_IMPEDANCE*frequency times the
    # right-going amplitude and its negative for a left-going amplitude.
    impedance = 1j * frequency * LEAD_IMPEDANCE
    boundary_matrix = np.array(
        [
            [impedance, stiffness * wave_number],
            [
                -stiffness * wave_number * sine - impedance * cosine,
                stiffness * wave_number * cosine - impedance * sine,
            ],
        ],
        dtype=complex,
    )
    boundary_data = np.array([2.0 * impedance * amplitude, 0.0], dtype=complex)
    a, b = np.linalg.solve(boundary_matrix, boundary_data)
    displacement, strain = _wave_integrals(a, b, wave_number, length)
    return density, displacement, strain


def predict_at(experiments, host_stiffness):
    """Predict the cycle-averaged mechanical energy for each experiment."""
    result = []
    for experiment in experiments:
        _, displacement, strain = wave_data(experiment, host_stiffness)
        frequency = float(experiment["frequency"])
        mass = float(experiment["resonator_mass"])
        resonance = float(experiment["resonance_frequency"])
        stiffness = float(host_stiffness)

        if mass:
            resonator_ratio = resonance**2 / (resonance**2 - frequency**2)
        else:
            resonator_ratio = 0.0

        # For a phasor of real amplitude A, the cycle average of a quadratic
        # term is one quarter of its coefficient times |A|^2.  The host
        # kinetic term is separated from the resonator terms here because
        # the latter have a different displacement phasor.
        resonator_kinetic = mass * frequency**2 * abs(resonator_ratio) ** 2
        spring = mass * resonance**2 * abs(resonator_ratio - 1.0) ** 2
        energy = (
            frequency**2 * displacement
            + resonator_kinetic * displacement
            + spring * displacement
            + stiffness * strain
        ) / 4.0
        result.append(float(np.real(energy)))
    return np.asarray(result, dtype=float)


def _objective(records, host_stiffness):
    """Weighted Gaussian least-squares objective for one stiffness value."""
    experiments = [record["input"] for record in records]
    observed = np.asarray([record["value"] for record in records], dtype=float)
    sigma = np.asarray([record["sigma"] for record in records], dtype=float)
    residual = (predict_at(experiments, host_stiffness) - observed) / sigma
    return float(residual @ residual)


def _minimize_interval(records, left, right, iterations=80):
    """Golden-section minimization on an interval containing one minimum."""
    golden = (np.sqrt(5.0) - 1.0) / 2.0
    x1 = right - golden * (right - left)
    x2 = left + golden * (right - left)
    f1 = _objective(records, x1)
    f2 = _objective(records, x2)
    for _ in range(iterations):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - golden * (right - left)
            f1 = _objective(records, x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + golden * (right - left)
            f2 = _objective(records, x2)
    return (left + right) / 2.0


class Model:
    def __init__(self):
        self.host_stiffness = None

    def fit(self, records):
        """Fit the common host stiffness using the calibration records.

        The search is one-dimensional and bounded by the physical interval
        in the specification.  A grid identifies the best basin, after which
        a bounded golden-section refinement gives sub-grid accuracy without
        requiring an optional optimization package.
        """
        records = list(records)
        if not records:
            raise ValueError("records must contain at least one calibration record")

        lower, upper = HOST_STIFFNESS_BOUNDS
        grid = np.linspace(lower, upper, 2001)
        objectives = np.asarray([_objective(records, value) for value in grid])
        best_index = int(np.argmin(objectives))

        if best_index == 0:
            estimate = lower
        elif best_index == len(grid) - 1:
            estimate = upper
        else:
            estimate = _minimize_interval(
                records, grid[best_index - 1], grid[best_index + 1]
            )

        self.host_stiffness = float(np.clip(estimate, lower, upper))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.host_stiffness)
