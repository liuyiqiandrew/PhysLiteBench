"""Prediction model for the finite-contact thermal apparatus.

The implementation uses a Galerkin representation of the backward generator
in products of normalized Hermite polynomials.  The motor sign is retained as
an additional two-state coordinate.  This is important: it is a persistent
state, rather than an instantaneous random choice of flow velocity.
"""

from functools import lru_cache

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.laguerre import laggauss
from scipy.linalg import solve


RATE = 3.6
FIELDS = (
    "angular_speed",
    "flip_rate",
    "temperature_a",
    "temperature_b",
    "flux_fraction",
    "relaxation_rate",
)


def normalized_hermites(z, degree):
    """Values of probabilists' normalized Hermites at ``z``."""
    z = np.asarray(z)
    h = np.zeros((z.size, degree + 1))
    h[:, 0] = 1.0
    if degree:
        h[:, 1] = z
    for n in range(1, degree):
        h[:, n + 1] = (z * h[:, n] - np.sqrt(n) * h[:, n - 1]) / np.sqrt(n + 1)
    return h


@lru_cache(maxsize=128)
def collision_weak_matrix(degree, variance=1.6, quadrature=None):
    """Galerkin matrix for ``abs(v-g) * (swap - identity)``.

    Under the reference product Gaussian, t=(v+g)/sqrt(2) and
    z=(v-g)/sqrt(2) are independent N(0, variance) variables.  In the
    Laguerre variable y=z**2/(2*variance), the factor ``abs(v-g)`` cancels
    the square-root Jacobian.  Thus the Laguerre weight has no extra factor
    of y; retaining one (as the original stub did) gives the wrong collision
    operator for nonconstant polynomials.
    """
    q = int(quadrature or (2 * degree + 6))
    th, tw = hermgauss(q)
    yl, yw = laggauss(q)

    t = np.sqrt(2.0 * variance) * th[None, :]
    blocks = []
    swapped = []
    weights = []
    for sign in (-1.0, 1.0):
        z = sign * np.sqrt(2.0 * variance * yl[:, None])
        v = (t + z) / np.sqrt(2.0)
        g = (t - z) / np.sqrt(2.0)
        hv = normalized_hermites(v.ravel() / np.sqrt(variance), degree)
        hg = normalized_hermites(g.ravel() / np.sqrt(variance), degree)
        b = (hv[:, :, None] * hg[:, None, :]).reshape(q * q, -1)
        bs = (hg[:, :, None] * hv[:, None, :]).reshape(q * q, -1)
        blocks.append(b)
        swapped.append(bs)
        # Hermite and Laguerre quadratures each carry a 1/sqrt(pi)
        # normalization.  The transformed |v-g| measure contributes
        # sqrt(variance), with one equal contribution for each sign.
        weights.append((np.sqrt(variance) / np.pi * yw[:, None] * tw[None, :]).ravel())

    b = np.concatenate(blocks)
    jump = np.concatenate(swapped) - b
    w = np.concatenate(weights)
    return b.T @ (w[:, None] * jump)


def _basis_matrices(degree, variance, omega):
    """Multiplication and derivative matrices on (s, v, g) polynomials."""
    width = degree + 1
    size = 2 * width * width

    def ix(k, n, m):
        return (k * width + n) * width + m

    G = np.zeros((size, size))
    D = np.zeros((size, size))
    S = np.zeros((size, size))
    G2 = np.zeros((size, size))
    for k in range(2):
        for n in range(width):
            for m in range(width):
                j = ix(k, n, m)
                S[ix(1 - k, n, m), j] = 1.0
                if m:
                    D[ix(k, n, m - 1), j] = np.sqrt(m / variance)
                if m + 1 < width:
                    G[ix(k, n, m + 1), j] = np.sqrt(variance * (m + 1))
                if m:
                    G[ix(k, n, m - 1), j] = np.sqrt(variance * m)
                G2[j, j] = variance * (2 * m + 1)
                if m >= 2:
                    G2[ix(k, n, m - 2), j] = variance * np.sqrt(m * (m - 1))
                if m + 2 < width:
                    G2[ix(k, n, m + 2), j] = variance * np.sqrt((m + 1) * (m + 2))

    R = G - omega * S
    R2 = G2 - 2.0 * omega * (S @ G) + omega * omega * np.eye(size)
    return R, R2, D, S


@lru_cache(maxsize=1024)
def _operator_bundle(
    angular_speed,
    flip_rate,
    temperature_a,
    temperature_b,
    flux_fraction,
    relaxation_rate,
    drag,
    degree,
):
    """Build the generator, heat operators, and stationary left vector."""
    omega = float(angular_speed)
    flip = float(flip_rate)
    ta = float(temperature_a)
    tb = float(temperature_b)
    eta = float(flux_fraction)
    lam = float(relaxation_rate)
    gamma = float(drag)
    variance = max(ta, tb)
    width = degree + 1
    block_size = width * width
    size = 2 * block_size

    def ix(k, n, m):
        return (k * width + n) * width + m

    L = np.zeros((size, size))
    C = collision_weak_matrix(degree, variance)

    for k in range(2):
        for n in range(width):
            for m in range(width):
                j = ix(k, n, m)
                L[j, j] -= gamma * n + lam * m + (2.0 * flip if k else 0.0)
                if n >= 2:
                    L[ix(k, n - 2, m), j] += gamma * (ta / variance - 1.0) * np.sqrt(n * (n - 1))
                if m >= 2:
                    L[ix(k, n, m - 2), j] += lam * (tb / variance - 1.0) * np.sqrt(m * (m - 1))
                if m >= 1:
                    L[ix(1 - k, n, m - 1), j] += lam * omega * np.sqrt(m / variance)

        lo = k * block_size
        hi = lo + block_size
        swap = np.zeros((block_size, block_size))
        for n in range(width):
            for m in range(width):
                swap[m * width + n, n * width + m] = 1.0
        L[lo:hi, lo:hi] += RATE * (1.0 - eta) * (swap - np.eye(block_size))
        L[lo:hi, lo:hi] += RATE * eta * C

    R, R2, D, S = _basis_matrices(degree, variance, omega)
    # For dQ = h dt + c dW, h=lam*(g-s*omega)^2-lam*Tb and
    # c*dW shares the g Wiener process.  The first tilted operator is
    # h - 2*lam*Tb*(g-s*omega)*d_g; the quadratic operator is c^2.
    B1 = lam * (R2 - tb * np.eye(size)) - 2.0 * lam * tb * (R @ D)
    B2 = 2.0 * lam * tb * R2
    del S

    system = L.T.copy()
    target = np.zeros(size)
    target[0] = 1.0
    system[0] = 0.0
    system[0, 0] = 1.0
    p = solve(system, target, check_finite=False)
    return L, B1, B2, p


def _cumulants(bundle, detector_second_moment=1.0):
    """Cumulant rates from the principal eigenvalue of the heat tilt."""
    L, B1, B2, p = bundle
    size = len(p)
    one = np.zeros(size)
    one[0] = 1.0
    mu2 = float(detector_second_moment)

    c1 = float(p @ B1[:, 0])
    system = L.copy()
    system[0] = p
    rhs1 = c1 * one - B1[:, 0]
    rhs1[0] = 0.0
    r1 = solve(system, rhs1, check_finite=False)

    c2 = float(p @ (mu2 * B2[:, 0] + 2.0 * B1 @ r1))
    rhs2 = c2 * one + 2.0 * c1 * r1 - 2.0 * B1 @ r1 - mu2 * B2[:, 0]
    rhs2[0] = 0.0
    r2 = solve(system, rhs2, check_finite=False)

    c3 = float(p @ (3.0 * B1 @ r2 + 3.0 * mu2 * B2 @ r1))
    return dict(mean=c1, noise=c2, third_cumulant=c3)


def _degree_for(eta, statistic):
    if float(eta) == 0.0:
        return 8
    # Use one common degree for all three readouts so a mixed prediction list
    # can reuse the same stationary generator bundle.  Degree 32 is close to
    # the converged value even at the largest allowed temperatures and flux.
    return 32


@lru_cache(maxsize=4096)
def readout(
    angular_speed,
    flip_rate,
    temperature_a,
    temperature_b,
    flux_fraction,
    relaxation_rate,
    statistic,
    drag,
):
    if statistic not in ("mean", "noise", "third_cumulant"):
        raise ValueError("unknown calorimeter statistic")
    e = dict(
        angular_speed=angular_speed,
        flip_rate=flip_rate,
        temperature_a=temperature_a,
        temperature_b=temperature_b,
        flux_fraction=flux_fraction,
        relaxation_rate=relaxation_rate,
    )
    if statistic == "mean" and float(flux_fraction) == 0.0 and float(angular_speed) == 0.0:
        return calibration_mean(e, drag)
    degree = _degree_for(flux_fraction, statistic)
    bundle = _operator_bundle(
        float(angular_speed), float(flip_rate), float(temperature_a), float(temperature_b),
        float(flux_fraction), float(relaxation_rate), float(drag), degree,
    )
    return _cumulants(bundle)[statistic]


def calibration_mean(e, drag):
    """Exact mean for the calibration preparation (constant collision clock)."""
    lam = float(e["relaxation_rate"])
    ta = float(e["temperature_a"])
    tb = float(e["temperature_b"])
    gamma = float(drag)
    return gamma * lam * RATE / (2.0 * gamma * lam + RATE * (gamma + lam)) * (ta - tb)


def fit_drag(records):
    """Weighted least-squares fit of the common laboratory drag."""
    if not records:
        raise ValueError("at least one calibration record is required")

    def prediction(gamma):
        return np.array([calibration_mean(r["input"], gamma) for r in records])

    y = np.array([float(r["value"]) for r in records])
    sigma = np.array([float(r.get("sigma", 1.0)) for r in records])
    if np.any(sigma <= 0) or not np.all(np.isfinite(y)):
        raise ValueError("calibration values and uncertainties must be finite")
    for r in records:
        x = r["input"]
        if x.get("angular_speed") != 0 or x.get("flux_fraction") != 0:
            raise ValueError("records are not the documented calibration preparation")
    grid = np.linspace(0.4, 1.1, 2001)
    losses = np.array([np.sum(((prediction(g) - y) / sigma) ** 2) for g in grid])
    gamma = float(grid[np.argmin(losses)])
    i = int(np.argmin(losses))
    if 0 < i < len(grid) - 1:
        x = grid[i - 1:i + 2]
        z = losses[i - 1:i + 2]
        coef = np.polyfit(x, z, 2)
        if coef[0] > 0:
            gamma = float(np.clip(-coef[1] / (2.0 * coef[0]), 0.4, 1.1))
    return gamma


@lru_cache(maxsize=2048)
def angular_readouts(
    temperature_a,
    temperature_b,
    relaxation_rate,
    flux_fraction,
    contrast,
    harmonic,
    drag,
    flip_rate=1.0,
):
    """Rapid-phase approximation for the optional angular detector."""
    del harmonic  # in the supplied rapid independent-phase limit
    bundle = _operator_bundle(
        0.0, float(flip_rate), float(temperature_a), float(temperature_b),
        float(flux_fraction), float(relaxation_rate), float(drag),
        _degree_for(flux_fraction, "third_cumulant"),
    )
    return _cumulants(bundle, detector_second_moment=1.0 + float(contrast) ** 2 / 2.0)


def predict_at(experiments, drag):
    if drag is None:
        raise RuntimeError("Model.fit must be called before Model.predict")
    result = []
    for e in experiments:
        statistic = e.get("statistic", "mean")
        contrast = float(e.get("detector_contrast", 0.0))
        if contrast:
            if float(e["angular_speed"]) != 0.0:
                raise ValueError("angularly gated preparations require zero flow target")
            values = angular_readouts(
                float(e["temperature_a"]), float(e["temperature_b"]),
                float(e["relaxation_rate"]), float(e["flux_fraction"]),
                contrast, int(e.get("detector_harmonic", 1)), float(drag),
                float(e["flip_rate"]),
            )
            if statistic not in values:
                raise ValueError("unknown calorimeter statistic")
            result.append(values[statistic])
        else:
            result.append(readout(*(float(e[f]) for f in FIELDS), statistic, float(drag)))
    return np.asarray(result, dtype=float)


class Model:
    def __init__(self):
        self.drag = None

    def fit(self, records):
        self.drag = fit_drag(records)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.drag)
