"""Stationary small-mass calorimetry of switched thermal contacts.

Use u = sqrt(m) v and expand the stationary velocity moments in sqrt(m).
Only moments through degree four are needed: leading even moments, first-order
odd moments, and the second-order corrections to populations and covariances.
The valve generator is part of the fast operator, just like velocity damping.
"""

from functools import lru_cache

import numpy as np


INPUT_FIELDS = (
    'force_x', 'force_y', 'drag_ratio', 'magnetic', 'temperature',
    'contrast', 'wavenumber', 'bath_contrast', 'switch_rate',
)


@lru_cache(None)
def spatial_grid(points):
    """Fourier collocation in temperature phase (use an odd grid size)."""
    phase = np.arange(points) * (2 * np.pi / points)
    modes = np.fft.fftfreq(points, 1 / points)
    derivative = np.fft.ifft(
        1j * modes[:, None] * np.fft.fft(np.eye(points), axis=0), axis=0,
    ).real
    return phase, derivative


def contact_parameters(e, friction):
    """Return temperature factors, drags, and the forward valve generator."""
    rate = e['switch_rate']
    if 'valve_cycle' not in e:
        labels = np.array([1., -1.])
        q = np.array([[-rate, rate], [rate, -rate]])
        drag_scale = np.ones(2)
    else:
        labels = np.array([-1., 0., 1.])
        bias = e['valve_cycle']
        q = np.zeros((3, 3))
        for s in range(3):
            q[(s + 1) % 3, s] = rate * (1 + bias)
            q[(s - 1) % 3, s] = rate * (1 - bias)
            q[s, s] = -2 * rate
        drag_scale = 1 + e.get('contact_drag_contrast', 0.) * labels
    gx = friction * drag_scale
    return 1 + e['bath_contrast'] * labels, gx, gx * e['drag_ratio'], q


def moment_operator(degree, gx, gy, magnetic, q):
    """Fast drift/switching matrix for moments u_x**p u_y**n."""
    count = len(gx)
    basis = [(p, degree - p) for p in range(degree, -1, -1)]
    lookup = {pair: i for i, pair in enumerate(basis)}
    a = np.kron(q, np.eye(degree + 1))
    for s in range(count):
        offset = s * (degree + 1)
        for i, (p, n) in enumerate(basis):
            row = offset + i
            a[row, row] -= p * gx[s] + n * gy[s]
            if p:
                a[row, offset + lookup[p - 1, n + 1]] += p * magnetic
            if n:
                a[row, offset + lookup[p + 1, n - 1]] -= n * magnetic
    return basis, a


def solve_moments(degree, lower, temperature, gx, gy, magnetic, q,
                  derivative=None, force=(0., 0.), prior=None, higher=None):
    """Solve one order of the stationary moment hierarchy.

    Noise couples to degree n-2 at the same expansion order. Force and spatial
    transport couple to degrees n-1 and n+1 at the preceding order.
    """
    count, points = temperature.shape
    basis, a = moment_operator(degree, gx, gy, magnetic, q)
    source = np.zeros((count, degree + 1, points))
    for s in range(count):
        for i, (p, n) in enumerate(basis):
            if p >= 2:
                source[s, i] += (
                    p * (p - 1) * gx[s] * temperature[s] * lower[p - 2, n][s]
                )
            if n >= 2:
                source[s, i] += (
                    n * (n - 1) * gy[s] * temperature[s] * lower[p, n - 2][s]
                )
            if prior is not None:
                if p:
                    source[s, i] += p * force[0] * prior[p - 1, n][s]
                if n:
                    source[s, i] += n * force[1] * prior[p, n - 1][s]
            if higher is not None:
                source[s, i] -= derivative @ higher[p + 1, n][s]
    value = np.linalg.solve(a, -source.reshape(count * (degree + 1), points))
    value = value.reshape(count, degree + 1, points)
    return {pair: value[:, i] for i, pair in enumerate(basis)}


def limiting_state(e, friction, points=129):
    """Compute the local excess, with density normalized in profile phase.

    The uniform background is exactly b/m. Its local leading counterpart is
    b*rho/m, so subtracting the *actual* residence also cancels b times the
    unknown order-m correction to the total positional density. We can thus
    choose that correction to be zero. The individual contact corrections
    still obey Q z = d(j_s)/dx and must be retained.
    """
    phase, d = spatial_grid(points)
    d = d * e['wavenumber']
    dx = 2 * np.pi / points
    factors, gx, gy, q = contact_parameters(e, friction)
    count = len(gx)
    force = (e['force_x'], e['force_y'])
    magnetic = e['magnetic']
    local_t = e['temperature'] * (1 + e['contrast'] * np.cos(phase))
    temperatures = factors[:, None] * local_t

    # Fast covariance per unit position density and base temperature.
    unit0 = {(0, 0): np.ones((count, 1)) / count}
    unit2 = solve_moments(2, unit0, factors[:, None], gx, gy, magnetic, q)
    background = float(np.sum(
        gx * (unit2[2, 0][:, 0] / factors - 1 / count)
        + gy * (unit2[0, 2][:, 0] / factors - 1 / count)
    ))
    _, a1 = moment_operator(1, gx, gy, magnetic, q)
    force_source = np.tile(np.array(force) / count, count)
    diff_source = np.column_stack((unit2[2, 0][:, 0], unit2[1, 1][:, 0])).ravel()
    velocity = -np.linalg.solve(a1, force_source).reshape(count, 2).sum(axis=0)
    diffusion = -np.linalg.solve(a1, diff_source).reshape(count, 2).sum(axis=0)

    # The total x current is V_x*rho - D_x*d(T*rho)/dx, and is constant.
    operator = -d @ (
        velocity[0] * np.eye(points) - diffusion[0] * d * local_t[None, :]
    )
    operator[-1] = dx
    rhs = np.zeros(points)
    rhs[-1] = 1.
    density = np.linalg.solve(operator, rhs)
    zeroth = {(0, 0): np.tile(density / count, (count, 1))}
    second = solve_moments(2, zeroth, temperatures, gx, gy, magnetic, q)
    fourth = solve_moments(4, second, temperatures, gx, gy, magnetic, q)
    first = solve_moments(
        1, {}, temperatures, gx, gy, magnetic, q, d, force, zeroth, second,
    )
    third = solve_moments(
        3, first, temperatures, gx, gy, magnetic, q, d, force, second, fourth,
    )

    # Invert Q on the zero-sum subspace; the removed stationary component is
    # precisely the total-density correction canceled by the background.
    divergence = first[1, 0] @ d.T
    divergence -= divergence.mean(axis=0)
    z = np.linalg.solve(q - np.ones((count, count)) / count, divergence)
    z -= z.mean(axis=0)
    density_correction = {(0, 0): z}
    second_correction = solve_moments(
        2, density_correction, temperatures, gx, gy, magnetic, q,
        d, force, first, third,
    )
    local = np.sum(
        gx[:, None] * (second_correction[2, 0] / temperatures - z)
        + gy[:, None] * (second_correction[0, 2] / temperatures - z), axis=0,
    ) - background * z.sum(axis=0)
    return dict(
        value=float(dx * local.sum()), local=local, phase=phase, density=density,
        diffusion=diffusion, velocity=velocity, dx=dx, first=first, second=second,
        density_correction=density_correction, second_correction=second_correction,
        temperatures=temperatures, background=background,
    )


@lru_cache(4096)
def entropy_rate(force_x, force_y, drag_ratio, magnetic, temperature, contrast,
                 wavenumber, bath_contrast, switch_rate, friction,
                 detector_phase=None, valve_cycle=None, contact_drag_contrast=0.):
    # Identical contacts at zero field reduce to one bath. All rates scale as
    # 1/friction, including spatially weighted rates; the valve is irrelevant.
    if bath_contrast == 0. and magnetic == 0. and valve_cycle is None and friction != 1.:
        return entropy_rate(
            force_x, force_y, drag_ratio, magnetic, temperature, contrast,
            wavenumber, bath_contrast, switch_rate, 1., detector_phase,
        ) / friction
    e = dict(zip(INPUT_FIELDS, (
        force_x, force_y, drag_ratio, magnetic, temperature, contrast,
        wavenumber, bath_contrast, switch_rate,
    )))
    if valve_cycle is not None:
        e.update(valve_cycle=valve_cycle, contact_drag_contrast=contact_drag_contrast)
    state = limiting_state(e, friction)
    weight = (1. if detector_phase is None else
              (1 + np.cos(state['phase'] - detector_phase)) / 2)
    return float(state['dx'] * np.sum(weight * state['local']))


def coefficients(experiments, friction=1.):
    return np.array([
        entropy_rate(
            *(e[f] for f in INPUT_FIELDS), float(friction),
            e.get('detector_phase'), e.get('valve_cycle'),
            e.get('contact_drag_contrast', 0.),
        )
        for e in experiments
    ], dtype=float)


class Model:
    def __init__(self):
        self.friction = None

    def fit(self, records):
        """Fit the shared friction by Gaussian weighted least squares.

        The documented calibration has identical baths and zero magnetic
        field, so its predictions are c/friction. Fit the reciprocal friction
        analytically and constrain it to the specified physical interval.
        """
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')
        inputs = [record['input'] for record in records]
        c = coefficients(inputs, friction=1.)
        values = np.array([record['value'] for record in records], dtype=float)
        sigma = np.array([record['sigma'] for record in records], dtype=float)
        if not (np.isfinite(values).all() and np.isfinite(sigma).all()
                and (sigma > 0).all()):
            raise ValueError('Calibration values must be finite and sigma positive.')
        design, target = c / sigma, values / sigma
        information = design @ design
        if information <= 0:
            raise ValueError('Calibration does not identify friction.')
        reciprocal = np.clip((design @ target) / information, 1 / 1.6, 1 / .7)
        self.friction = float(1 / reciprocal)
        return self

    def predict(self, experiments):
        if self.friction is None:
            raise RuntimeError('Fit the model before predicting.')
        return coefficients(experiments, self.friction)
