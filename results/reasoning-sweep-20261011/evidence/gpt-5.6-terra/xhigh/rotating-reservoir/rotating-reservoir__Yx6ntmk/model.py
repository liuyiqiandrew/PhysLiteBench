"""Stationary calorimeter statistics for the finite-contact rotor model.

The calculation is a Galerkin solution of the backward generator. Hermite
polynomials make the OU and heat terms sparse; the non-polynomial collision
operator is evaluated once by Gaussian quadrature. The motor remains its
two-state Markov chain, rather than being replaced by an averaged flow.
"""

from functools import lru_cache

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.laguerre import laggauss
from scipy.linalg import lu_factor, lu_solve, solve
from scipy.optimize import minimize_scalar


RATE = 3.6
FIELDS = (
    "angular_speed", "flip_rate", "temperature_a", "temperature_b",
    "flux_fraction", "relaxation_rate",
)


def normalized_hermites(x, degree):
    """Normalized probabilists' Hermite polynomials through ``degree``."""
    x = np.asarray(x)
    ans = np.empty((x.size, degree + 1), dtype=float)
    ans[:, 0] = 1.0
    if degree:
        ans[:, 1] = x
    for n in range(1, degree):
        ans[:, n + 1] = (x * ans[:, n] - np.sqrt(n) * ans[:, n - 1]) / np.sqrt(n + 1)
    return ans


@lru_cache(maxsize=8)
def _unit_collision_matrix(degree, quadrature=None):
    """Weak matrix for ``abs(v-g)`` under independent unit Gaussians.

    The collision part is ``RATE * eta * C``. A reference Gaussian with
    variance q merely multiplies C by sqrt(q), so this costly integral is
    independent of experiment temperature.
    """
    qn = int(quadrature or (2 * degree + 6))
    gh_x, gh_w = hermgauss(qn)
    lg_x, lg_w = laggauss(qn)
    t = np.sqrt(2.0) * gh_x[None, :]
    functions = []
    exchanged = []
    weights = []
    for sign in (-1.0, 1.0):
        z = sign * np.sqrt(2.0 * lg_x[:, None])
        v = (t + z) / np.sqrt(2.0)
        g = (t - z) / np.sqrt(2.0)
        hv = normalized_hermites(v.ravel(), degree)
        hg = normalized_hermites(g.ravel(), degree)
        functions.append((hv[:, :, None] * hg[:, None, :]).reshape(qn * qn, -1))
        exchanged.append((hg[:, :, None] * hv[:, None, :]).reshape(qn * qn, -1))
        weights.append((lg_w[:, None] * gh_w[None, :] / np.pi).ravel())
    basis = np.concatenate(functions, axis=0)
    jump = np.concatenate(exchanged, axis=0) - basis
    weight = np.concatenate(weights)
    matrix = basis.T @ (weight[:, None] * jump)
    return (matrix + matrix.T) * 0.5


def collision_weak_matrix(degree, variance=1.0, quadrature=None):
    """Compatibility wrapper for the relative-speed collision matrix."""
    return np.sqrt(float(variance)) * _unit_collision_matrix(int(degree), quadrature)


def _indices(degree, total_degree):
    width = degree + 1
    pairs = [(n, m) for n in range(width) for m in range(width)
             if not total_degree or n + m <= degree]
    return pairs, {pair: i for i, pair in enumerate(pairs)}


def _operators(degree, variance, total_degree=True):
    """Polynomial multiplication/differentiation operators and pair basis."""
    pairs, lookup = _indices(degree, total_degree)
    size = len(pairs)
    v = np.zeros((size, size))
    g = np.zeros((size, size))
    dg = np.zeros((size, size))
    g2 = np.zeros((size, size))

    def put(matrix, pair, col, value):
        row = lookup.get(pair)
        if row is not None:
            matrix[row, col] += value

    for col, (n, m) in enumerate(pairs):
        if n:
            put(v, (n - 1, m), col, np.sqrt(variance * n))
        put(v, (n + 1, m), col, np.sqrt(variance * (n + 1)))
        if m:
            put(g, (n, m - 1), col, np.sqrt(variance * m))
            put(dg, (n, m - 1), col, np.sqrt(m / variance))
        put(g, (n, m + 1), col, np.sqrt(variance * (m + 1)))
        # Direct construction retains the virtual degree m+1 contribution to
        # the diagonal which g@g would lose at the spectral boundary.
        put(g2, (n, m), col, variance * (2 * m + 1))
        if m >= 2:
            put(g2, (n, m - 2), col, variance * np.sqrt(m * (m - 1)))
        put(g2, (n, m + 2), col, variance * np.sqrt((m + 1) * (m + 2)))
    return pairs, lookup, v, g, dg, g2


def _generator_and_heat(e, drag, degree, total_degree=True):
    """Backward generator and the first/two heat-tilt derivatives."""
    ta = float(e["temperature_a"])
    tb = float(e["temperature_b"])
    omega = float(e["angular_speed"])
    lam = float(e["relaxation_rate"])
    eta = float(e["flux_fraction"])
    flip = float(e["flip_rate"])
    variance = max(ta, tb, 0.25)
    pairs, lookup, v, g, dg, g2 = _operators(degree, variance, total_degree)
    size = len(pairs)
    identity = np.eye(size)
    base = np.zeros((size, size))

    for col, (n, m) in enumerate(pairs):
        base[col, col] -= drag * n + lam * m
        if n >= 2:
            row = lookup.get((n - 2, m))
            if row is not None:
                base[row, col] += drag * (ta / variance - 1.0) * np.sqrt(n * (n - 1))
        if m >= 2:
            row = lookup.get((n, m - 2))
            if row is not None:
                base[row, col] += lam * (tb / variance - 1.0) * np.sqrt(m * (m - 1))
        row = lookup.get((m, n))
        if row is not None:
            base[row, col] += RATE * (1.0 - eta)
        base[col, col] -= RATE * (1.0 - eta)

    full_width = degree + 1
    selected = np.fromiter((n * full_width + m for n, m in pairs), dtype=int)
    collision = _unit_collision_matrix(degree)[np.ix_(selected, selected)]
    base += RATE * eta * np.sqrt(variance) * collision

    # In the even/signed motor basis a test function is f0(v,g)+s*f1(v,g).
    flow = lam * omega * dg
    generator = np.block([[base, flow], [flow, base - 2.0 * flip * identity]])

    # dQ=a dt+b dW. The heat tilt has a+b*sigma_g*d/dg at first order.
    x_dg = g @ dg
    a0 = lam * (g2 + (omega * omega - tb) * identity) - 2.0 * lam * tb * x_dg
    a1 = -2.0 * lam * omega * g + 2.0 * lam * tb * omega * dg
    b20 = 2.0 * lam * tb * (g2 + omega * omega * identity)
    b21 = -4.0 * lam * tb * omega * g
    first = np.block([[a0, a1], [a1, a0]])
    second = np.block([[b20, b21], [b21, b20]])
    return generator, first, second


def _stationary_cumulants(generator, first, second, detector_contrast=0.0):
    """First three derivatives of the tilted principal eigenvalue."""
    count = generator.shape[0]
    one = np.zeros(count)
    one[0] = 1.0
    stat = generator.T.copy()
    stat[0, :] = 0.0
    stat[0, 0] = 1.0
    rhs = np.zeros(count)
    rhs[0] = 1.0
    p = solve(stat, rhs, assume_a="gen", check_finite=False)

    # The documented gated approximation independently and rapidly mixes the
    # detector phase: E[w]=1 and E[w^2]=1+contrast^2/2.
    second = second * (1.0 + 0.5 * float(detector_contrast) ** 2)
    mean = float(p @ first @ one)
    response = generator.copy()
    response[0, :] = p
    lu, piv = lu_factor(response, check_finite=False)

    r1_rhs = mean * one - first @ one
    r1_rhs[0] = 0.0
    r1 = lu_solve((lu, piv), r1_rhs, check_finite=False)
    noise = float(p @ (2.0 * first @ r1 + second @ one))

    r2_rhs = noise * one + 2.0 * mean * r1 - 2.0 * first @ r1 - second @ one
    r2_rhs[0] = 0.0
    r2 = lu_solve((lu, piv), r2_rhs, check_finite=False)
    third = float(p @ (3.0 * first @ r2 + 3.0 * second @ r1))
    return {"mean": mean, "noise": noise, "third_cumulant": third}


def kinetic_statistics(e, drag, degree=None):
    """All three stationary cumulant rates for one preparation."""
    eta = float(e["flux_fraction"])
    if degree is None:
        degree = 10 if eta == 0.0 else 40
    generator, first, second = _generator_and_heat(e, float(drag), int(degree), True)
    return _stationary_cumulants(generator, first, second, e.get("detector_contrast", 0.0))


def calibration_mean(e, drag):
    """Exact mean for the supplied zero-flow, constant-clock calibration."""
    lam = float(e["relaxation_rate"])
    return (float(drag) * lam * RATE /
            (2.0 * float(drag) * lam + RATE * (float(drag) + lam)) *
            (float(e["temperature_a"]) - float(e["temperature_b"])))


def fit_drag(records):
    """Weighted one-parameter fit, respecting the known [.4, 1.1] interval."""
    if not records:
        raise ValueError("at least one calibration record is required")

    def objective(drag):
        total = 0.0
        for record in records:
            sigma = float(record["sigma"])
            if not sigma > 0.0:
                raise ValueError("calibration uncertainties must be positive")
            residual = (calibration_mean(record["input"], drag) - float(record["value"])) / sigma
            total += residual * residual
        return total

    result = minimize_scalar(objective, bounds=(0.4, 1.1), method="bounded",
                             options={"xatol": 1e-13})
    return float(result.x)


@lru_cache(maxsize=2048)
def _readout_cached(angular_speed, flip_rate, temperature_a, temperature_b,
                    flux_fraction, relaxation_rate, statistic, contrast,
                    harmonic, drag):
    if statistic not in {"mean", "noise", "third_cumulant"}:
        raise ValueError("unknown calorimeter statistic")
    if contrast != 0.0 and angular_speed != 0.0:
        raise ValueError("Angularly gated preparations require zero flow target.")
    if harmonic not in (1, 2, 3):
        raise ValueError("detector_harmonic must be 1, 2, or 3")
    e = dict(zip(FIELDS, (angular_speed, flip_rate, temperature_a, temperature_b,
                          flux_fraction, relaxation_rate)))
    e["detector_contrast"] = contrast
    if statistic == "mean" and flux_fraction == 0.0 and angular_speed == 0.0 and contrast == 0.0:
        return float(calibration_mean(e, drag))
    return _statistics_cached(angular_speed, flip_rate, temperature_a, temperature_b,
                              flux_fraction, relaxation_rate, contrast, drag)[statistic]


@lru_cache(maxsize=1024)
def _statistics_cached(angular_speed, flip_rate, temperature_a, temperature_b,
                       flux_fraction, relaxation_rate, contrast, drag):
    """Cache all readouts together, so a mixed readout list shares a solve."""
    e = dict(zip(FIELDS, (angular_speed, flip_rate, temperature_a, temperature_b,
                          flux_fraction, relaxation_rate)))
    e["detector_contrast"] = contrast
    return kinetic_statistics(e, drag)


def predict_at(experiments, drag):
    if drag is None:
        raise RuntimeError("fit must be called before predict")
    output = []
    for experiment in experiments:
        values = tuple(float(experiment[field]) for field in FIELDS)
        statistic = experiment.get("statistic", "mean")
        contrast = float(experiment.get("detector_contrast", 0.0))
        harmonic = int(experiment.get("detector_harmonic", 1))
        output.append(_readout_cached(*values, statistic, contrast, harmonic, float(drag)))
    return np.asarray(output, dtype=float)


class Model:
    def __init__(self):
        self.drag = None

    def fit(self, records):
        self.drag = fit_drag(records)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.drag)
