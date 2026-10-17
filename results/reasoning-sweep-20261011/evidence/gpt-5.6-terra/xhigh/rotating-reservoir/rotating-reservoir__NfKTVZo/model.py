"""Stationary calorimeter model for the finite retained gas contact.

The implementation uses a Galerkin representation in normalized Hermite
polynomials.  This is particularly convenient here: both OU generators are
banded, whereas the only non-polynomial part of the generator (the
relative-speed collision clock) can be integrated accurately with a
Gauss--Hermite/Gauss--Laguerre product rule.
"""

from functools import lru_cache

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.laguerre import laggauss
from scipy.linalg import solve


RATE = 3.6
FIELDS = (
    "angular_speed", "flip_rate", "temperature_a", "temperature_b",
    "flux_fraction", "relaxation_rate",
)

# A fixed reference measure means that all experiments share the expensive
# collision matrix.  It is only a basis scale, not a physical approximation.
REFERENCE_VARIANCE = 1.0
DEFAULT_DEGREE = 24


def normalized_hermites(z, degree):
    """Orthonormal probabilists' Hermite polynomials through ``degree``."""
    z = np.asarray(z)
    h = np.empty((z.size, degree + 1), dtype=float)
    h[:, 0] = 1.0
    if degree:
        h[:, 1] = z
    for n in range(1, degree):
        h[:, n + 1] = (z * h[:, n] - np.sqrt(n) * h[:, n - 1]) / np.sqrt(n + 1)
    return h


@lru_cache(maxsize=8)
def _basis(degree):
    """Total-degree basis and lookup table for one velocity pair."""
    pairs = tuple((n, m) for n in range(degree + 1) for m in range(degree + 1 - n))
    return pairs, {pair: i for i, pair in enumerate(pairs)}


@lru_cache(maxsize=8)
def collision_weak_matrix(degree, variance=REFERENCE_VARIANCE):
    """Projected generator for ``abs(v-g) * (swap - identity)``.

    In centre/relative coordinates the absolute value is removed by a
    Laguerre rule for the squared relative velocity.  The integrand after
    summing its two signs is polynomial, so the chosen orders integrate all
    products represented by this total-degree space exactly (to roundoff).
    """
    pairs, _ = _basis(degree)
    size = len(pairs)
    q = 2 * degree + 4
    nodes_h, weights_h = hermgauss(q)
    nodes_l, weights_l = laggauss(q)

    t = np.sqrt(2.0 * variance) * nodes_h[None, :]
    blocks = []
    swapped_blocks = []
    weights = []
    for sign in (-1.0, 1.0):
        z = sign * np.sqrt(2.0 * variance * nodes_l[:, None])
        v = (t + z) / np.sqrt(2.0)
        g = (t - z) / np.sqrt(2.0)
        hv = normalized_hermites(v.ravel() / np.sqrt(variance), degree)
        hg = normalized_hermites(g.ravel() / np.sqrt(variance), degree)
        value = np.empty((q * q, size), dtype=float)
        after = np.empty_like(value)
        for j, (n, m) in enumerate(pairs):
            value[:, j] = hv[:, n] * hg[:, m]
            after[:, j] = hv[:, m] * hg[:, n]
        blocks.append(value)
        swapped_blocks.append(after)
        # Includes |v-g| = sqrt(2)|z| and both normalized Gaussian weights.
        weights.append((np.sqrt(variance) / np.pi * weights_l[:, None] * weights_h[None, :]).ravel())

    value = np.concatenate(blocks, axis=0)
    jump = np.concatenate(swapped_blocks, axis=0) - value
    weight = np.concatenate(weights)
    matrix = value.T @ (weight[:, None] * jump)
    # The exact operator is self-adjoint and negative semidefinite under the
    # isotropic reference Gaussian. Symmetrizing removes quadrature roundoff.
    return 0.5 * (matrix + matrix.T)


def _linear_operators(e, drag, degree=DEFAULT_DEGREE):
    """Return generator and heat-tilt derivative operators.

    Matrices act on coefficients of functions (the backward generator).
    There are two blocks, one for each prescribed motor label.  Keeping this
    label explicit is essential because the flow target is ``s*omega``.
    """
    ta = float(e["temperature_a"])
    tb = float(e["temperature_b"])
    omega = float(e["angular_speed"])
    relaxation = float(e["relaxation_rate"])
    eta = float(e["flux_fraction"])
    flip = float(e["flip_rate"])
    variance = REFERENCE_VARIANCE
    pairs, lookup = _basis(degree)
    one_size = len(pairs)
    size = 2 * one_size
    matrix = np.zeros((size, size), dtype=float)
    xg = np.zeros_like(matrix)
    dg = np.zeros_like(matrix)
    g2 = np.zeros_like(matrix)
    gd = np.zeros_like(matrix)

    def ind(block, n, m):
        return block * one_size + lookup[(n, m)]

    # OU pieces, the constant collision channel, and motor transitions.
    for block, sign in enumerate((1.0, -1.0)):
        target = sign * omega
        for n, m in pairs:
            j = ind(block, n, m)
            matrix[j, j] -= drag * n + relaxation * m + RATE * (1.0 - eta) + flip
            matrix[ind(1 - block, n, m), j] += flip
            matrix[ind(block, m, n), j] += RATE * (1.0 - eta)
            if n >= 2:
                matrix[ind(block, n - 2, m), j] += drag * (ta / variance - 1.0) * np.sqrt(n * (n - 1.0))
            if m >= 2:
                matrix[ind(block, n, m - 2), j] += relaxation * (tb / variance - 1.0) * np.sqrt(m * (m - 1.0))
            if m >= 1:
                matrix[ind(block, n, m - 1), j] += relaxation * target * np.sqrt(m / variance)

            # Multiplication and differentiation in the gas coordinate.
            if m >= 1:
                xg[ind(block, n, m - 1), j] += np.sqrt(variance * m)
                dg[ind(block, n, m - 1), j] += np.sqrt(m / variance)
                if m >= 2:
                    gd[ind(block, n, m - 2), j] += np.sqrt(m * (m - 1.0))
                gd[j, j] += m
            if (n, m + 1) in lookup:
                xg[ind(block, n, m + 1), j] += np.sqrt(variance * (m + 1.0))
            g2[j, j] += variance * (2.0 * m + 1.0)
            if m >= 2:
                g2[ind(block, n, m - 2), j] += variance * np.sqrt(m * (m - 1.0))
            if (n, m + 2) in lookup:
                g2[ind(block, n, m + 2), j] += variance * np.sqrt((m + 1.0) * (m + 2.0))

    if eta:
        collision = RATE * eta * collision_weak_matrix(degree, variance)
        matrix[:one_size, :one_size] += collision
        matrix[one_size:, one_size:] += collision

    first = np.zeros_like(matrix)
    second = np.zeros_like(matrix)
    for block, sign in enumerate((1.0, -1.0)):
        sl = slice(block * one_size, (block + 1) * one_size)
        target = sign * omega
        relative_square = g2[sl, sl] - 2.0 * target * xg[sl, sl] + target * target * np.eye(one_size)
        # For exp(k Q), A1 = a + b*sigma*d_g and A2 = b^2.
        first[sl, sl] = (
            relaxation * (relative_square - tb * np.eye(one_size))
            - 2.0 * relaxation * tb * (gd[sl, sl] - target * dg[sl, sl])
        )
        second[sl, sl] = 2.0 * relaxation * tb * relative_square

    one = np.zeros(size)
    one[0] = one[one_size] = 1.0
    return matrix, first, second, one


def _stationary_left(matrix, one):
    """Stationary expectation functional for a backward-generator matrix."""
    system = matrix.T.copy()
    rhs = np.zeros(len(one))
    system[0, :] = one
    rhs[0] = 1.0
    return solve(system, rhs, check_finite=False, assume_a="gen")


def _poisson_solve(matrix, stationary, rhs):
    """Solve M r=rhs with stationary(r)=0; rhs must have zero mean."""
    system = matrix.copy()
    source = np.array(rhs, dtype=float, copy=True)
    system[0, :] = stationary
    source[0] = 0.0
    return solve(system, source, check_finite=False, assume_a="gen")


def kinetic_statistics(e, drag, degree=DEFAULT_DEGREE, detector_contrast=0.0):
    """Mean and the first two centered cumulant rates of the heat signal."""
    matrix, first, second, one = _linear_operators(e, float(drag), int(degree))
    stationary = _stationary_left(matrix, one)
    mean = float(stationary @ (first @ one))

    r1 = _poisson_solve(matrix, stationary, mean * one - first @ one)

    # The documented gated preparation uses independently rapidly mixed
    # detector phase.  Thus E[w]=1 in A1, while its instantaneous quadratic
    # variation has E[w^2]=1+contrast^2/2.  The harmonic drops out in this
    # limiting detector model.
    shot_scale = 1.0 + 0.5 * float(detector_contrast) ** 2
    second_gated = shot_scale * second
    noise = float(stationary @ (second_gated @ one + 2.0 * first @ r1))
    r2 = _poisson_solve(
        matrix,
        stationary,
        noise * one + 2.0 * mean * r1 - 2.0 * first @ r1 - second_gated @ one,
    )
    third = float(stationary @ (3.0 * first @ r2 + 3.0 * second_gated @ r1))
    return {"mean": mean, "noise": noise, "third_cumulant": third}


def calibration_mean(e, drag):
    """Exact mean when the collision channel is the constant-clock channel."""
    relaxation = float(e["relaxation_rate"])
    conductance = float(drag) * relaxation * RATE / (
        2.0 * float(drag) * relaxation + RATE * (float(drag) + relaxation)
    )
    return conductance * (float(e["temperature_a"]) - float(e["temperature_b"]))


def fit_drag(records):
    """Weighted least-squares calibration of the one common drag parameter."""
    if not records:
        raise ValueError("at least one calibration record is required")
    contrasts = []
    values = []
    for record in records:
        e = record["input"]
        if float(e["flux_fraction"]) != 0.0 or float(e["angular_speed"]) != 0.0:
            raise ValueError("calibration records must have zero flow and zero flux fraction")
        sigma = float(record["sigma"])
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("calibration sigma values must be positive")
        contrasts.append((float(e["temperature_a"]) - float(e["temperature_b"])) / sigma)
        values.append(float(record["value"]) / sigma)
    contrasts = np.asarray(contrasts)
    values = np.asarray(values)
    denom = float(contrasts @ contrasts)
    if denom == 0.0:
        raise ValueError("calibration temperatures do not identify drag")
    conductance = float(contrasts @ values / denom)
    # Invert C=d*l*R / (2dl + R(d+l)).  The clip enforces the apparatus prior
    # and also protects against a noisy, nonphysical unconstrained estimate.
    relaxation = float(records[0]["input"]["relaxation_rate"])
    drag = conductance * relaxation * RATE / (
        relaxation * RATE - conductance * (2.0 * relaxation + RATE)
    )
    return float(np.clip(drag, 0.4, 1.1))


@lru_cache(maxsize=1024)
def _statistics_cached(angular_speed, flip_rate, temperature_a, temperature_b,
                       flux_fraction, relaxation_rate, drag,
                       detector_contrast):
    e = dict(zip(FIELDS, (angular_speed, flip_rate, temperature_a, temperature_b,
                          flux_fraction, relaxation_rate)))
    values = kinetic_statistics(e, drag, detector_contrast=detector_contrast)
    return values["mean"], values["noise"], values["third_cumulant"]


@lru_cache(maxsize=2048)
def _readout_cached(angular_speed, flip_rate, temperature_a, temperature_b,
                    flux_fraction, relaxation_rate, statistic, drag,
                    detector_contrast):
    if statistic not in ("mean", "noise", "third_cumulant"):
        raise ValueError("unknown calorimeter statistic")
    e = dict(zip(FIELDS, (angular_speed, flip_rate, temperature_a, temperature_b,
                          flux_fraction, relaxation_rate)))
    # This special case is exact, fast, and is also the calibration law.
    if statistic == "mean" and flux_fraction == 0.0 and angular_speed == 0.0:
        return float(calibration_mean(e, drag))
    values = _statistics_cached(angular_speed, flip_rate, temperature_a, temperature_b,
                                flux_fraction, relaxation_rate, drag,
                                detector_contrast)
    return values[("mean", "noise", "third_cumulant").index(statistic)]


def predict_at(experiments, drag):
    if drag is None:
        raise RuntimeError("call fit(records) before predict(experiments)")
    result = []
    for experiment in experiments:
        contrast = float(experiment.get("detector_contrast", 0.0))
        angular_speed = float(experiment["angular_speed"])
        if contrast and angular_speed != 0.0:
            raise ValueError("Angularly gated preparations require zero flow target.")
        # detector_harmonic is validated for the documented API even though it
        # cancels from the independently mixed-phase detector approximation.
        harmonic = int(experiment.get("detector_harmonic", 1))
        if harmonic not in (1, 2, 3):
            raise ValueError("detector_harmonic must be 1, 2, or 3")
        result.append(_readout_cached(
            *(float(experiment[field]) for field in FIELDS),
            experiment.get("statistic", "mean"), float(drag), contrast,
        ))
    return np.asarray(result, dtype=float)


class Model:
    def __init__(self):
        self.drag = None

    def fit(self, records):
        self.drag = fit_drag(records)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.drag)
