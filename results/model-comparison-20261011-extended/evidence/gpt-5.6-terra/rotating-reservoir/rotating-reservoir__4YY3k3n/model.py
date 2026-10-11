"""Stationary calorimeter statistics for the finite-contact rotor model.

The calculation uses a Galerkin expansion in normalized Hermite polynomials.
It keeps the two motor states explicitly.  This is particularly convenient
here: the OU terms and the swap part of a collision are sparse, while the
relative-speed part of the collision clock is evaluated by Gaussian
quadrature once for each basis scale.
"""
from functools import lru_cache

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.laguerre import laggauss
from scipy.linalg import solve
from scipy.optimize import minimize_scalar


RATE = 3.6
FIELDS = (
    "angular_speed", "flip_rate", "temperature_a", "temperature_b",
    "flux_fraction", "relaxation_rate",
)


def normalized_hermites(z, degree):
    """Values of the orthonormal probabilists' Hermite polynomials."""
    z = np.asarray(z)
    values = np.empty((z.size, degree + 1), dtype=float)
    values[:, 0] = 1.0
    if degree:
        values[:, 1] = z
    for n in range(1, degree):
        values[:, n + 1] = (z * values[:, n] - np.sqrt(n) * values[:, n - 1]) / np.sqrt(n + 1)
    return values


@lru_cache(maxsize=24)
def collision_weak_matrix(degree, variance):
    """Weak matrix of ``|v-g| (swap - identity)``.

    The reference measure is two independent N(0, variance) variables.  In
    center/relative coordinates, Laguerre quadrature integrates the absolute
    relative velocity without ever approximating its cusp.
    """
    order = 2 * degree + 5
    nodes_t, weights_t = hermgauss(order)
    nodes_y, weights_y = laggauss(order)
    t = np.sqrt(2.0 * variance) * nodes_t[None, :]
    bases, swaps, weights = [], [], []
    for sign in (-1.0, 1.0):
        z = sign * np.sqrt(2.0 * variance * nodes_y[:, None])
        v = (t + z) / np.sqrt(2.0)
        g = (t - z) / np.sqrt(2.0)
        hv = normalized_hermites(v.ravel() / np.sqrt(variance), degree)
        hg = normalized_hermites(g.ravel() / np.sqrt(variance), degree)
        bases.append((hv[:, :, None] * hg[:, None, :]).reshape(order * order, -1))
        swaps.append((hg[:, :, None] * hv[:, None, :]).reshape(order * order, -1))
        weights.append((np.sqrt(variance) / np.pi * weights_y[:, None] * weights_t[None, :]).ravel())
    basis = np.concatenate(bases)
    jump = np.concatenate(swaps) - basis
    weight = np.concatenate(weights)
    matrix = basis.T @ (weight[:, None] * jump)
    return (matrix + matrix.T) * 0.5


def _indices(degree):
    pairs = [(n, m) for n in range(degree + 1) for m in range(degree + 1 - n)]
    return pairs, {pair: i for i, pair in enumerate(pairs)}


def _operators(e, drag, degree):
    """Backward generator and the first two tilted-generator derivatives."""
    ta, tb = float(e["temperature_a"]), float(e["temperature_b"])
    omega, lam = float(e["angular_speed"]), float(e["relaxation_rate"])
    eta, flip = float(e["flux_fraction"]), float(e["flip_rate"])
    # This is only a basis scale, not a physical approximation to a bath.
    variance = max(ta, tb, 0.75 + omega * omega)
    pairs, lookup = _indices(degree)
    count, size = len(pairs), 2 * len(pairs)
    loc = lambda state, pair: state * count + lookup[pair]
    L = np.zeros((size, size), dtype=float)
    Dg, G, G2 = np.zeros_like(L), np.zeros_like(L), np.zeros_like(L)

    full_width = degree + 1
    full_index = lambda n, m: n * full_width + m
    collision_full = collision_weak_matrix(degree, float(variance))
    full_keep = [full_index(*x) for x in pairs]
    collision = collision_full[np.ix_(full_keep, full_keep)]

    for state, sign in enumerate((-1.0, 1.0)):
        other = 1 - state
        for j, (n, m) in enumerate(pairs):
            col = state * count + j
            L[col, col] -= drag * n + lam * m + RATE * (1.0 - eta) + flip
            L[loc(state, (m, n)), col] += RATE * (1.0 - eta)
            L[other * count + j, col] += flip
            if n >= 2:
                L[loc(state, (n - 2, m)), col] += drag * (ta / variance - 1.0) * np.sqrt(n * (n - 1))
            if m >= 2:
                L[loc(state, (n, m - 2)), col] += lam * (tb / variance - 1.0) * np.sqrt(m * (m - 1))
            if m >= 1:
                L[loc(state, (n, m - 1)), col] += lam * sign * omega * np.sqrt(m / variance)
                Dg[loc(state, (n, m - 1)), col] = np.sqrt(m / variance)
                G[loc(state, (n, m - 1)), col] = np.sqrt(variance * m)
            if m + 1 <= degree - n:
                G[loc(state, (n, m + 1)), col] = np.sqrt(variance * (m + 1))
            G2[col, col] = variance * (2 * m + 1)
            if m >= 2:
                G2[loc(state, (n, m - 2)), col] = variance * np.sqrt(m * (m - 1))
            if m + 2 <= degree - n:
                G2[loc(state, (n, m + 2)), col] = variance * np.sqrt((m + 1) * (m + 2))

    for state in range(2):
        block = slice(state * count, (state + 1) * count)
        L[block, block] += RATE * eta * collision

    identity = np.eye(size)
    target = np.concatenate((np.full(count, -omega), np.full(count, omega)))
    U = np.diag(target)
    relative_g = G - U
    relative_g2 = G2 - 2.0 * U @ G + U @ U
    A1 = 2.0 * lam * tb * relative_g @ Dg - lam * (relative_g2 - tb * identity)
    A2 = 2.0 * lam * tb * relative_g2

    stationary_system = L.T.copy()
    stationary_rhs = np.zeros(size)
    stationary_system[0, :] = 0.0
    stationary_system[0, 0] = stationary_system[0, count] = 1.0
    stationary_rhs[0] = 1.0
    p = solve(stationary_system, stationary_rhs, check_finite=False)
    one = np.zeros(size)
    one[0] = one[count] = 1.0
    return L, A1, A2, p, one


def kinetic_statistics(e, drag, detector_contrast=0.0, statistic="third_cumulant"):
    """Requested stationary calorimeter cumulant rate.

    A constant collision clock preserves polynomial degree, so degree six is
    exact for these three readouts.  The relative-speed clock creates a cusp
    at v=g; the larger truncations below were selected by direct convergence
    checks at the edges of the documented parameter box.
    """
    if float(e["flux_fraction"]) == 0.0:
        degree = 6
    elif statistic == "third_cumulant":
        degree = 32
    else:
        degree = 24
    L, A1, A2, p, one = _operators(e, float(drag), degree)
    # The rapid independently mixed detector phase changes only E[w^2].
    A2 = A2 * (1.0 + 0.5 * float(detector_contrast) ** 2)
    lambda1 = float(p @ (A1 @ one))
    system = L.copy()
    system[0, :] = p
    rhs1 = lambda1 * one - A1 @ one
    rhs1[0] = 0.0
    r1 = solve(system, rhs1, check_finite=False)
    if statistic == "mean":
        return {"mean": -lambda1}
    lambda2 = float(p @ (A2 @ one + 2.0 * A1 @ r1))
    if statistic == "noise":
        return {"noise": lambda2}
    rhs2 = lambda2 * one + 2.0 * lambda1 * r1 - 2.0 * A1 @ r1 - A2 @ one
    rhs2[0] = 0.0
    r2 = solve(system, rhs2, check_finite=False)
    lambda3 = float(p @ (3.0 * A1 @ r2 + 3.0 * A2 @ r1))
    return {"third_cumulant": -lambda3}


def calibration_mean(e, drag):
    """Exact zero-flow, constant-clock stationary heat current."""
    lam = float(e["relaxation_rate"])
    conductance = drag * lam * RATE / (2.0 * drag * lam + RATE * (drag + lam))
    return float(conductance * (float(e["temperature_a"]) - float(e["temperature_b"])))


@lru_cache(maxsize=4096)
def _readout(values, statistic, drag, contrast):
    e = dict(zip(FIELDS, values))
    if statistic not in {"mean", "noise", "third_cumulant"}:
        raise ValueError("unknown calorimeter statistic")
    if contrast == 0.0 and statistic == "mean" and e["flux_fraction"] == 0.0 and e["angular_speed"] == 0.0:
        return calibration_mean(e, drag)
    return float(kinetic_statistics(e, drag, contrast, statistic)[statistic])


def predict_at(experiments, drag):
    if drag is None:
        raise RuntimeError("fit(records) must be called before predict(experiments)")
    output = []
    for experiment in experiments:
        contrast = float(experiment.get("detector_contrast", 0.0))
        if contrast and float(experiment["angular_speed"]) != 0.0:
            raise ValueError("Angularly gated preparations require zero flow target.")
        values = tuple(float(experiment[name]) for name in FIELDS)
        output.append(_readout(values, str(experiment.get("statistic", "mean")), float(drag), contrast))
    return np.asarray(output, dtype=float)


def fit_drag(records):
    """Weighted maximum-likelihood estimate for the common drag."""
    if not records:
        raise ValueError("at least one calibration record is required")
    values = np.asarray([float(record["value"]) for record in records])
    sigmas = np.asarray([float(record["sigma"]) for record in records])
    if np.any(sigmas <= 0.0):
        raise ValueError("calibration uncertainties must be positive")

    def objective(drag):
        predicted = np.asarray([calibration_mean(record["input"], drag) for record in records])
        return float(np.sum(((predicted - values) / sigmas) ** 2))

    optimum = minimize_scalar(objective, bounds=(0.4, 1.1), method="bounded", options={"xatol": 1e-12})
    return float(optimum.x)


class Model:
    def __init__(self):
        self.drag = None

    def fit(self, records):
        self.drag = fit_drag(records)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.drag)
