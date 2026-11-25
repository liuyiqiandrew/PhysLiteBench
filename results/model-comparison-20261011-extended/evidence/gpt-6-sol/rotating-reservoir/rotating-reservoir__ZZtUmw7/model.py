"""Stationary heat cumulants for the retained, colliding gas contact."""

from functools import lru_cache

import numpy as np
from numpy.polynomial.laguerre import laggauss
from scipy.linalg import solve


COLLISION_RATE = 3.6
FIELDS = (
    'angular_speed', 'flip_rate', 'temperature_a', 'temperature_b',
    'flux_fraction', 'relaxation_rate',
)
STATISTICS = ('mean', 'noise', 'third_cumulant')


def calibration_mean(experiment, drag):
    """Exact zero-flow, constant-clock stationary heat current."""
    rate = experiment['relaxation_rate']
    conductance = drag * rate * COLLISION_RATE / (
        2 * drag * rate + COLLISION_RATE * (drag + rate)
    )
    return conductance * (experiment['temperature_a'] - experiment['temperature_b'])


def fit_drag(records):
    # Calibration has a shared relaxation rate, zero flow and constant clock.
    rate = float(records[0]['input']['relaxation_rate'])
    temperature_difference = np.array([
        (r['input']['temperature_a'] - r['input']['temperature_b']) / r['sigma']
        for r in records
    ], dtype=float)
    heat = np.array([r['value'] / r['sigma'] for r in records], dtype=float)
    conductance = float(temperature_difference @ heat /
                        (temperature_difference @ temperature_difference))
    drag = conductance * rate * COLLISION_RATE / (
        rate * COLLISION_RATE - conductance * (2 * rate + COLLISION_RATE)
    )
    return float(np.clip(drag, 0.4, 1.1))


def _basis(degree):
    pairs = [(a, total - a) for total in range(degree + 1)
             for a in range(total + 1)]
    return pairs, {pair: j for j, pair in enumerate(pairs)}


@lru_cache(maxsize=32)
def _collision_integral(degree, variance):
    """Matrix of |z| in odd Hermites for z=(v-g)/sqrt(2)."""
    # After y=z²/(2 variance), |z| times the Gaussian weight is
    # exp(-y) dy. Products of two odd Hermites become polynomials in y.
    roots, weights = laggauss(degree + 2)
    x = np.sqrt(2 * roots)
    h = np.empty((len(roots), degree + 1))
    h[:, 0] = 1
    if degree:
        h[:, 1] = x
    for n in range(1, degree):
        h[:, n + 1] = (x * h[:, n] - np.sqrt(n) * h[:, n - 1]) / np.sqrt(n + 1)
    odd = h[:, 1::2]
    return np.sqrt(2 * variance / np.pi) * (odd.T * weights) @ odd


def _operators(experiment, drag, degree):
    """Backward generator and the first two heat-tilt derivatives.

    The Hermite coordinates are z=(v-g)/sqrt(2), t=(v+g)/sqrt(2).
    For nonzero flow, a polynomial of odd total degree carries one factor
    of the motor label s. This is the invariant reflection sector.
    """
    ta = experiment['temperature_a']
    tb = experiment['temperature_b']
    rate = experiment['relaxation_rate']
    flux = experiment['flux_fraction']
    omega = experiment['angular_speed']
    # At zero flow the motor is dynamically irrelevant. In particular, its
    # parity sector cannot be used for angle Fourier modes, which mix parity.
    flip = experiment['flip_rate'] if omega != 0 else 0.0
    variance = max(ta, tb)

    # Work two degrees higher before restriction: multiplication by g²
    # must retain paths that briefly leave the requested polynomial space.
    pairs, lookup = _basis(degree + 2)
    big = len(pairs)
    small = (degree + 1) * (degree + 2) // 2
    z = np.zeros((big, big))
    t = np.zeros_like(z)
    dz = np.zeros_like(z)
    dt = np.zeros_like(z)
    root_variance = np.sqrt(variance)
    for j, (a, b) in enumerate(pairs):
        if a:
            z[lookup[a - 1, b], j] = root_variance * np.sqrt(a)
            dz[lookup[a - 1, b], j] = np.sqrt(a) / root_variance
        if b:
            t[lookup[a, b - 1], j] = root_variance * np.sqrt(b)
            dt[lookup[a, b - 1], j] = np.sqrt(b) / root_variance
        if a + b < degree + 2:
            z[lookup[a + 1, b], j] = root_variance * np.sqrt(a + 1)
            t[lookup[a, b + 1], j] = root_variance * np.sqrt(b + 1)
    inv_root_two = 1 / np.sqrt(2)
    v = (t + z) * inv_root_two
    g = (t - z) * inv_root_two
    dv = (dt + dz) * inv_root_two
    dg = (dt - dz) * inv_root_two
    identity = np.eye(big)

    # In the reflection sector, the matrix g represents s*g whenever
    # it changes polynomial parity; likewise dg represents s*dg.
    relative_square = g @ g - 2 * omega * g + omega * omega * identity
    generator = (-drag * (v @ dv) - rate * (g @ dg)
                 + drag * ta * (dv @ dv) + rate * tb * (dg @ dg)
                 + rate * omega * dg)
    first = (2 * rate * tb * (g @ dg - omega * dg)
             - rate * (relative_square - tb * identity))
    second = 2 * rate * tb * relative_square
    generator = generator[:small, :small].copy()
    first = first[:small, :small].copy()
    second = second[:small, :small].copy()
    v = v[:small, :small].copy()

    integral = _collision_integral(degree, variance)
    for col, (a, b) in enumerate(pairs[:small]):
        if a % 2:
            generator[col, col] -= 2 * COLLISION_RATE * (1 - flux)
            for row_a in range(1, degree - b + 1, 2):
                row = lookup[row_a, b]
                generator[row, col] -= (2 * COLLISION_RATE * flux * np.sqrt(2)
                                        * integral[(row_a - 1) // 2, (a - 1) // 2])
        if (a + b) % 2:
            generator[col, col] -= 2 * flip
    return generator, v, first, second


def _stationary(generator):
    system = generator.T.copy()
    system[0] = 0
    system[0, 0] = 1
    target = np.zeros(len(generator))
    target[0] = 1
    return solve(system, target, assume_a='gen', check_finite=False)


def _centered_solve(generator, stationary, rhs):
    system = generator.copy()
    system[0] = stationary
    rhs = rhs.copy()
    rhs[0] = 0
    return solve(system, rhs, assume_a='gen', check_finite=False)


def _cumulants(experiment, drag, degree):
    generator, velocity, first, second = _operators(experiment, drag, degree)
    stationary = _stationary(generator)
    constant = np.zeros(len(generator))
    constant[0] = 1
    contrast = experiment.get('detector_contrast', 0.0)
    harmonic = experiment.get('detector_harmonic', 1)
    first_constant = first[:, 0]
    second_constant = second[:, 0]
    c1 = float(stationary @ first_constant)
    r10 = _centered_solve(generator, stationary, c1 * constant - first_constant)

    if contrast:
        mode = generator.astype(complex) + 1j * harmonic * velocity
        r11 = solve(mode, -(contrast / 2) * first_constant,
                    assume_a='gen', check_finite=False)
        first_mode = np.real(first @ r11)
    else:
        r11 = None
        first_mode = 0
    c2 = float(stationary @ ((1 + contrast**2 / 2) * second_constant
                              + 2 * (first @ r10 + contrast * first_mode)))
    rhs20 = (c2 * constant + 2 * c1 * r10
             - 2 * (first @ r10 + contrast * first_mode)
             - (1 + contrast**2 / 2) * second_constant)
    r20 = _centered_solve(generator, stationary, rhs20)

    if contrast:
        rhs21 = (2 * c1 * r11 - contrast * (first @ r10)
                 - 2 * (first @ r11) - contrast * second_constant)
        r21 = solve(mode, rhs21, assume_a='gen', check_finite=False)
        third_modes = contrast * np.real(first @ r21) + 2 * contrast * np.real(second @ r11)
    else:
        third_modes = 0
    c3 = float(3 * stationary @ (first @ r20 + third_modes
                                 + (1 + contrast**2 / 2) * (second @ r10)))
    return {'mean': -c1, 'noise': c2, 'third_cumulant': -c3}


@lru_cache(maxsize=512)
def _readouts(settings, drag):
    experiment = dict(settings)
    if experiment.get('detector_contrast', 0.0) and experiment['angular_speed'] != 0:
        raise ValueError('Angularly gated preparations require zero flow target.')
    # Flux collisions couple odd relative Hermites of all degrees. With a
    # constant clock, only angle gating needs more than six degrees because
    # the Fourier generator contains multiplication by velocity.
    if experiment['flux_fraction']:
        degree = 40
    elif experiment.get('detector_contrast', 0.0):
        degree = 28
    else:
        degree = 6
    return _cumulants(experiment, drag, degree)


class Model:
    def __init__(self):
        self.drag = None

    def fit(self, records):
        self.drag = fit_drag(records)
        return self

    def predict(self, experiments):
        if self.drag is None:
            raise ValueError('Model must be fitted before prediction.')
        outputs = []
        for experiment in experiments:
            statistic = experiment.get('statistic', 'mean')
            if statistic not in STATISTICS:
                raise ValueError('Unknown calorimeter statistic: ' + str(statistic))
            if (statistic == 'mean' and experiment['flux_fraction'] == 0
                    and experiment['angular_speed'] == 0):
                outputs.append(calibration_mean(experiment, self.drag))
            else:
                settings = tuple((key, float(experiment[key])) for key in FIELDS)
                settings += (('detector_contrast', float(experiment.get('detector_contrast', 0.0))),
                             ('detector_harmonic', int(experiment.get('detector_harmonic', 1))))
                outputs.append(_readouts(settings, self.drag)[statistic])
        return np.asarray(outputs, dtype=float)
