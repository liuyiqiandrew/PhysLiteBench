"""Stationary small-mass calorimetry of switched thermal contacts.

The velocity and valve dynamics are fast on the small-mass time scale.  This
module solves the resulting (periodic) moment hierarchy through the order
which remains after the matched-background subtraction.  It is deliberately
implemented with NumPy only: the calibration is a one-dimensional weighted
least-squares problem and the spatial derivatives are Fourier derivatives.
"""

from functools import lru_cache

import numpy as np


INPUT_FIELDS = (
    "force_x", "force_y", "drag_ratio", "magnetic", "temperature",
    "contrast", "wavenumber", "bath_contrast", "switch_rate",
)


@lru_cache(maxsize=None)
def spatial_grid(points):
    """Return a periodic grid and its spectral first-derivative matrix."""
    phase = np.arange(points, dtype=float) * (2.0 * np.pi / points)
    modes = np.fft.fftfreq(points, 1.0 / points)
    # Column j is the derivative of the periodic cardinal function at j.
    derivative = np.fft.ifft(
        1j * modes[:, None] * np.fft.fft(np.eye(points), axis=0), axis=0
    ).real
    return phase, derivative


def contact_parameters(experiment, friction):
    """Fast valve generator and state-specific thermal drag parameters."""
    rate = float(experiment["switch_rate"])
    if "valve_cycle" not in experiment:
        # The generator acts on a column of state moments.
        labels = np.array([1.0, -1.0])
        generator = np.array([[-rate, rate], [rate, -rate]])
        drag_scale = np.ones(2)
    else:
        labels = np.array([-1.0, 0.0, 1.0])
        bias = float(experiment["valve_cycle"])
        generator = np.zeros((3, 3))
        for state in range(3):
            generator[(state + 1) % 3, state] = rate * (1.0 + bias)
            generator[(state - 1) % 3, state] = rate * (1.0 - bias)
            generator[state, state] = -2.0 * rate
        drag_scale = 1.0 + float(experiment.get("contact_drag_contrast", 0.0)) * labels

    temperature_scale = 1.0 + float(experiment["bath_contrast"]) * labels
    gamma_x = float(friction) * drag_scale
    gamma_y = gamma_x * float(experiment["drag_ratio"])
    return temperature_scale, gamma_x, gamma_y, generator


def moment_operator(degree, gamma_x, gamma_y, magnetic, generator):
    """Matrix for degree-``degree`` fast velocity moments.

    The ordering is (u_x**p u_y**n), p+n=degree, state by state.  Here
    u=sqrt(m)*v, so this matrix is independent of m.
    """
    count = len(gamma_x)
    basis = [(p, degree - p) for p in range(degree, -1, -1)]
    lookup = {pair: index for index, pair in enumerate(basis)}
    operator = np.kron(generator, np.eye(degree + 1))

    for state in range(count):
        for index, (p, n) in enumerate(basis):
            row = state * (degree + 1) + index
            operator[row, row] -= p * gamma_x[state] + n * gamma_y[state]
            if p:
                operator[row, state * (degree + 1) + lookup[p - 1, n + 1]] += p * magnetic
            if n:
                operator[row, state * (degree + 1) + lookup[p + 1, n - 1]] -= n * magnetic
    return basis, operator


def solve_moments(
    degree,
    lower,
    temperature,
    gamma_x,
    gamma_y,
    magnetic,
    generator,
    derivative=None,
    force=(0.0, 0.0),
    prior=None,
    higher=None,
):
    """Solve one degree of the fast-moment hierarchy.

    ``lower`` is the degree-2 lower moment at the same expansion order;
    ``prior`` and ``higher`` belong to the preceding expansion order.
    """
    count, points = temperature.shape
    basis, operator = moment_operator(degree, gamma_x, gamma_y, magnetic, generator)
    source = np.zeros((count, degree + 1, points))

    for state in range(count):
        for index, (p, n) in enumerate(basis):
            if p >= 2:
                source[state, index] += p * (p - 1) * gamma_x[state] * temperature[state] * lower[p - 2, n][state]
            if n >= 2:
                source[state, index] += n * (n - 1) * gamma_y[state] * temperature[state] * lower[p, n - 2][state]
            if prior is not None:
                if p:
                    source[state, index] += p * force[0] * prior[p - 1, n][state]
                if n:
                    source[state, index] += n * force[1] * prior[p, n - 1][state]
            if higher is not None:
                source[state, index] -= derivative @ higher[p + 1, n][state]

    values = np.linalg.solve(operator, -source.reshape(count * (degree + 1), points))
    values = values.reshape(count, degree + 1, points)
    return {pair: values[:, index] for index, pair in enumerate(basis)}


def _zero_sum_contact_correction(generator, rhs):
    """Solve Q z=rhs pointwise, selecting the solution with sum(z)=0.

    At second order, the state-resolved positional density has a determined
    component transverse to the stationary contact distribution.  Its common
    component is not needed: it cancels exactly against B_m R_m.
    """
    constrained = generator.copy()
    constrained[-1, :] = 1.0
    constrained_rhs = np.array(rhs, copy=True)
    constrained_rhs[-1, :] = 0.0
    return np.linalg.solve(constrained, constrained_rhs)


def limiting_state(experiment, friction, points=129):
    """Return the finite, background-subtracted local entropy coefficient."""
    phase, derivative_theta = spatial_grid(points)
    wave_number = int(experiment["wavenumber"])
    derivative = wave_number * derivative_theta  # d/dx when theta=k*x
    dx = 2.0 * np.pi / points

    temperature_scale, gamma_x, gamma_y, generator = contact_parameters(experiment, friction)
    count = len(gamma_x)
    force = (float(experiment["force_x"]), float(experiment["force_y"]))
    magnetic = float(experiment["magnetic"])
    local_temperature = float(experiment["temperature"]) * (
        1.0 + float(experiment["contrast"]) * np.cos(phase)
    )
    temperatures = temperature_scale[:, None] * local_temperature[None, :]

    # The leading contact probabilities are uniform for both valve designs.
    leading_uniform = {(0, 0): np.ones((count, 1)) / count}
    unit_second = solve_moments(
        2, leading_uniform, temperature_scale[:, None], gamma_x, gamma_y,
        magnetic, generator,
    )

    # Effective one-dimensional current: J = v*rho - D*d_x(T*rho).
    _, first_operator = moment_operator(1, gamma_x, gamma_y, magnetic, generator)
    force_source = np.tile(np.asarray(force) / count, count)
    gradient_source = np.column_stack(
        (unit_second[2, 0][:, 0], unit_second[1, 1][:, 0])
    ).ravel()
    velocity = -np.linalg.solve(first_operator, force_source).reshape(count, 2).sum(axis=0)
    diffusion = -np.linalg.solve(first_operator, gradient_source).reshape(count, 2).sum(axis=0)

    stationary_operator = -derivative @ (
        velocity[0] * np.eye(points) - diffusion[0] * derivative * local_temperature[None, :]
    )
    stationary_operator[-1, :] = dx
    stationary_rhs = np.zeros(points)
    stationary_rhs[-1] = 1.0
    density = np.linalg.solve(stationary_operator, stationary_rhs)

    zeroth = {(0, 0): np.tile(density / count, (count, 1))}
    second = solve_moments(2, zeroth, temperatures, gamma_x, gamma_y, magnetic, generator)
    fourth = solve_moments(4, second, temperatures, gamma_x, gamma_y, magnetic, generator)
    first = solve_moments(
        1, {}, temperatures, gamma_x, gamma_y, magnetic, generator,
        derivative, force, zeroth, second,
    )
    third = solve_moments(
        3, first, temperatures, gamma_x, gamma_y, magnetic, generator,
        derivative, force, second, fourth,
    )

    # Q M_0^(2)=d_x M_x^(1).  The discarded common component is precisely
    # the contribution removed by the measured weighted residence.
    density_correction = {
        (0, 0): _zero_sum_contact_correction(generator, first[1, 0] @ derivative.T)
    }
    second_correction = solve_moments(
        2, density_correction, temperatures, gamma_x, gamma_y, magnetic,
        generator, derivative, force, first, third,
    )

    # This is the O(m^0) term of sum_i gamma_i(<u_i^2>/T-<1>).
    # The O(1/m) common-density term has already cancelled against B_m R_m.
    local = np.sum(
        gamma_x[:, None] * (second_correction[2, 0] / temperatures - density_correction[0, 0])
        + gamma_y[:, None] * (second_correction[0, 2] / temperatures - density_correction[0, 0]),
        axis=0,
    )
    return {
        "value": float(dx * local.sum()),
        "local": local,
        "phase": phase,
        "density": density,
        "diffusion": diffusion,
        "velocity": velocity,
        "dx": dx,
    }


@lru_cache(maxsize=4096)
def entropy_rate(
    force_x, force_y, drag_ratio, magnetic, temperature, contrast, wavenumber,
    bath_contrast, switch_rate, friction, detector_phase=None, valve_cycle=None,
    contact_drag_contrast=0.0,
):
    """Limiting excess entropy rate for one hashable experiment description."""
    # In the calibration setting contacts are identical.  The stationary
    # density is friction independent and the finite entropy rate scales as
    # 1/friction, so reuse the unit-friction spectral solve.
    if bath_contrast == 0.0 and magnetic == 0.0 and valve_cycle is None and friction != 1.0:
        return entropy_rate(
            force_x, force_y, drag_ratio, magnetic, temperature, contrast,
            wavenumber, bath_contrast, switch_rate, 1.0, detector_phase,
        ) / friction

    experiment = dict(zip(
        INPUT_FIELDS,
        (force_x, force_y, drag_ratio, magnetic, temperature, contrast,
         wavenumber, bath_contrast, switch_rate),
    ))
    if valve_cycle is not None:
        experiment.update(
            valve_cycle=valve_cycle,
            contact_drag_contrast=contact_drag_contrast,
        )
    state = limiting_state(experiment, friction)
    if detector_phase is None:
        weight = 1.0
    else:
        weight = (1.0 + np.cos(state["phase"] - detector_phase)) / 2.0
    return float(state["dx"] * np.sum(weight * state["local"]))


def coefficients(experiments, friction=1.0):
    """Vectorized public prediction helper, retaining experiment order."""
    return np.asarray([
        entropy_rate(
            *(experiment[field] for field in INPUT_FIELDS),
            float(friction),
            experiment.get("detector_phase"),
            experiment.get("valve_cycle"),
            experiment.get("contact_drag_contrast", 0.0),
        )
        for experiment in experiments
    ], dtype=float)


class Model:
    def __init__(self):
        self.friction = None

    def fit(self, records):
        """Fit the shared friction by inverse-variance weighted least squares."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record["input"] for record in records]
        observed = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)
        if not (np.isfinite(observed).all() and np.isfinite(sigma).all() and np.all(sigma > 0.0)):
            raise ValueError("calibration values must be finite and sigmas must be positive")

        # Calibration is documented to use b=B=0 and identical contacts.  In
        # that case prediction a/f is exact, yielding a stable analytic fit.
        unit = coefficients(experiments, 1.0)
        inverse_friction = np.sum(unit * observed / sigma**2) / np.sum(unit**2 / sigma**2)
        if not np.isfinite(inverse_friction) or inverse_friction <= 0.0:
            raise ValueError("calibration records do not identify positive friction")
        self.friction = float(np.clip(1.0 / inverse_friction, 0.7, 1.6))
        return self

    def predict(self, experiments):
        if self.friction is None:
            raise RuntimeError("call fit(records) before predict(experiments)")
        return coefficients(experiments, self.friction)
