"""Stationary calorimeter statistics for the finite retained gas contact.

The calculation uses a Hermite--Galerkin representation of the backward
generator.  A gated detector is represented by Fourier modes of its actual
angle, whose generator is ``v d/dtheta``; its phase is not independently
refreshed.
"""
from functools import lru_cache

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.laguerre import laggauss
from scipy.linalg import lu_factor, lu_solve

RATE = 3.6
FIELDS = ("angular_speed", "flip_rate", "temperature_a", "temperature_b",
          "flux_fraction", "relaxation_rate")


def normalized_hermites(z, degree):
    z = np.asarray(z)
    h = np.zeros((z.size, degree + 1)); h[:, 0] = 1.0
    if degree: h[:, 1] = z
    for n in range(1, degree):
        h[:, n + 1] = (z * h[:, n] - np.sqrt(n) * h[:, n - 1]) / np.sqrt(n + 1)
    return h


@lru_cache(maxsize=64)
def collision_weak_matrix(degree, variance, quadrature=None):
    """Galerkin matrix of ``|v-g| (f(g,v)-f(v,g))``.

    Laguerre quadrature treats the relative-speed cusp directly.
    """
    q = quadrature or (2 * degree + 6)
    xh, wh = hermgauss(q); xl, wl = laggauss(q)
    centre = np.sqrt(2.0 * variance) * xh[None, :]
    values, swaps, weights = [], [], []
    for sign in (-1.0, 1.0):
        relative = sign * np.sqrt(2.0 * variance * xl[:, None])
        v = (centre + relative) / np.sqrt(2.0)
        g = (centre - relative) / np.sqrt(2.0)
        hv = normalized_hermites(v.ravel() / np.sqrt(variance), degree)
        hg = normalized_hermites(g.ravel() / np.sqrt(variance), degree)
        values.append((hv[:, :, None] * hg[:, None, :]).reshape(q * q, -1))
        swaps.append((hg[:, :, None] * hv[:, None, :]).reshape(q * q, -1))
        weights.append((np.sqrt(variance) / np.pi * wl[:, None] * wh[None, :]).ravel())
    value = np.concatenate(values)
    jump = np.concatenate(swaps) - value
    weight = np.concatenate(weights)
    return value.T @ (weight[:, None] * jump)


def _operators(e, drag, degree):
    ta, tb = float(e["temperature_a"]), float(e["temperature_b"])
    lam, eta = float(e["relaxation_rate"]), float(e["flux_fraction"])
    variance = max(ta, tb); width = degree + 1; size = width * width
    ind = lambda n, m: n * width + m
    L = np.zeros((size, size)); V = np.zeros_like(L); G = np.zeros_like(L)
    D = np.zeros_like(L); G2 = np.zeros_like(L)
    for n in range(width):
        for m in range(width):
            j = ind(n, m)
            L[j, j] -= drag * n + lam * m + RATE * (1.0 - eta)
            L[ind(m, n), j] += RATE * (1.0 - eta)
            if n >= 2:
                L[ind(n - 2, m), j] += drag * (ta / variance - 1.0) * np.sqrt(n * (n - 1))
            if m >= 2:
                L[ind(n, m - 2), j] += lam * (tb / variance - 1.0) * np.sqrt(m * (m - 1))
            if n: V[ind(n - 1, m), j] += np.sqrt(variance * n)
            if n < degree: V[ind(n + 1, m), j] += np.sqrt(variance * (n + 1))
            if m:
                G[ind(n, m - 1), j] += np.sqrt(variance * m)
                D[ind(n, m - 1), j] += np.sqrt(m / variance)
            if m < degree: G[ind(n, m + 1), j] += np.sqrt(variance * (m + 1))
            G2[j, j] += variance * (2 * m + 1)
            if m >= 2: G2[ind(n, m - 2), j] += variance * np.sqrt(m * (m - 1))
            if m + 2 < width: G2[ind(n, m + 2), j] += variance * np.sqrt((m + 1) * (m + 2))
    L += RATE * eta * collision_weak_matrix(degree, variance)
    return L, V, G, D, G2


def _constant(size):
    value = np.zeros(size); value[0] = 1.0
    return value


def _stationary(L, constant):
    system = L.T.copy(); rhs = np.zeros(len(constant))
    system[0, :] = constant; rhs[0] = 1.0
    return lu_solve(lu_factor(system, check_finite=False), rhs, check_finite=False)


def _poisson_solver(L, p):
    """Inverse of L on mean-zero functions, normalized by p r=0."""
    system = L.astype(complex, copy=True); system[0, :] = p
    factors = lu_factor(system, check_finite=False)
    def solve(rhs):
        rhs = np.asarray(rhs, dtype=complex).copy(); rhs[0] = 0.0
        return lu_solve(factors, rhs, check_finite=False)
    return solve


def _heat_operators(G, D, G2, lam, tb, target):
    identity = np.eye(len(G)); x = G - target * identity
    # These are d/dk and d2/dk2 of the backward tilted generator for exp(-kQ).
    return (2.0 * lam * tb * (x @ D) - lam * (x @ x - tb * identity),
            2.0 * lam * tb * (x @ x))


def _cumulants(L, p, constant, a1, a2, statistic):
    mu1 = float(np.real(p @ (a1 @ constant)))
    if statistic == "mean": return -mu1
    inverse = _poisson_solver(L, p)
    r1 = inverse(mu1 * constant - a1 @ constant)
    mu2 = float(np.real(p @ (a2 @ constant + 2.0 * a1 @ r1)))
    if statistic == "noise": return mu2
    r2 = inverse(mu2 * constant + 2.0 * mu1 * r1 - 2.0 * a1 @ r1 - a2 @ constant)
    mu3 = float(np.real(p @ (3.0 * a1 @ r2 + 3.0 * a2 @ r1)))
    return -mu3


def _ungated(e, drag, statistic):
    """Heat statistics retaining the switching motor state when flow is nonzero."""
    degree = 8 if float(e["flux_fraction"]) == 0.0 else 28
    L0, _V, G, D, G2 = _operators(e, drag, degree)
    omega, flip = float(e["angular_speed"]), float(e["flip_rate"])
    lam, tb = float(e["relaxation_rate"]), float(e["temperature_b"])
    n = len(L0)
    if omega == 0.0:
        constant = _constant(n); p = _stationary(L0, constant)
        a1, a2 = _heat_operators(G, D, G2, lam, tb, 0.0)
        return _cumulants(L0, p, constant, a1, a2, statistic)
    L = np.zeros((2 * n, 2 * n)); eye = np.eye(n)
    L[:n, :n] = L0 - flip * eye; L[n:, n:] = L0 - flip * eye
    L[:n, n:] = flip * eye; L[n:, :n] = flip * eye
    constant = np.zeros(2 * n); constant[0] = constant[n] = 1.0
    p = _stationary(L, constant)
    a1minus, a2minus = _heat_operators(G, D, G2, lam, tb, -omega)
    a1plus, a2plus = _heat_operators(G, D, G2, lam, tb, omega)
    a1 = np.zeros_like(L); a2 = np.zeros_like(L)
    a1[:n, :n], a1[n:, n:] = a1minus, a1plus
    a2[:n, :n], a2[n:, n:] = a2minus, a2plus
    return _cumulants(L, p, constant, a1, a2, statistic)


def _gated(e, drag, statistic):
    """Statistics of w(theta)Q with all Fourier modes required through order 3."""
    if float(e["angular_speed"]) != 0.0:
        raise ValueError("Angularly gated preparations require zero flow target.")
    contrast = float(e.get("detector_contrast", 0.0))
    if contrast == 0.0: return _ungated(e, drag, statistic)
    harmonic = int(e.get("detector_harmonic", 1))
    L, V, G, D, G2 = _operators(e, drag, 28)
    constant = _constant(len(L)); p = _stationary(L, constant)
    lam, tb = float(e["relaxation_rate"]), float(e["temperature_b"])
    a1, a2 = _heat_operators(G, D, G2, lam, tb, 0.0)
    mu1 = float(np.real(p @ (a1 @ constant)))
    if statistic == "mean": return -mu1
    w = {0: 1.0, -1: contrast / 2.0, 1: contrast / 2.0}
    w2 = {0: 1.0 + contrast * contrast / 2.0, -1: contrast, 1: contrast,
          -2: contrast * contrast / 4.0, 2: contrast * contrast / 4.0}
    inverse = {0: _poisson_solver(L, p)}
    for mode in (-2, -1, 1, 2):
        factors = lu_factor(L + 1j * mode * harmonic * V, check_finite=False)
        inverse[mode] = lambda rhs, factors=factors: lu_solve(factors, rhs, check_finite=False)
    r1 = {mode: inverse[mode]((mu1 * constant if mode == 0 else 0.0) -
                               w[mode] * (a1 @ constant)) for mode in (-1, 0, 1)}
    def wa1(mode, response):
        return sum(w.get(mode - j, 0.0) * (a1 @ value) for j, value in response.items())
    mu2 = float(np.real(p @ (w2[0] * (a2 @ constant) + 2.0 * wa1(0, r1))))
    if statistic == "noise": return mu2
    r2 = {}
    for mode in (-2, -1, 0, 1, 2):
        rhs = (mu2 * constant if mode == 0 else 0.0) + 2.0 * mu1 * r1.get(mode, 0.0)
        rhs = rhs - 2.0 * wa1(mode, r1) - w2[mode] * (a2 @ constant)
        r2[mode] = inverse[mode](rhs)
    mu3 = float(np.real(p @ (3.0 * wa1(0, r2) + 3.0 * sum(
        w2.get(-mode, 0.0) * (a2 @ value) for mode, value in r1.items()))))
    return -mu3


@lru_cache(maxsize=4096)
def _readout_cached(angular_speed, flip_rate, temperature_a, temperature_b,
                    flux_fraction, relaxation_rate, statistic, contrast, harmonic, drag):
    if statistic not in ("mean", "noise", "third_cumulant"):
        raise ValueError("unknown calorimeter statistic")
    e = dict(zip(FIELDS, (angular_speed, flip_rate, temperature_a, temperature_b,
                           flux_fraction, relaxation_rate)))
    e["detector_contrast"] = contrast; e["detector_harmonic"] = harmonic
    return float(_gated(e, drag, statistic) if contrast else _ungated(e, drag, statistic))


def predict_at(experiments, drag):
    if drag is None: raise RuntimeError("Call fit before predict.")
    output = []
    for e in experiments:
        output.append(_readout_cached(*(float(e[f]) for f in FIELDS), e.get("statistic", "mean"),
                                      float(e.get("detector_contrast", 0.0)),
                                      int(e.get("detector_harmonic", 1)), float(drag)))
    return np.asarray(output, dtype=float)


def calibration_mean(e, drag):
    lam = float(e["relaxation_rate"])
    return drag * lam * RATE / (2.0 * drag * lam + RATE * (drag + lam)) * (
        float(e["temperature_a"]) - float(e["temperature_b"]))


def fit_drag(records):
    if not records: raise ValueError("at least one calibration record is required")
    lam = float(records[0]["input"]["relaxation_rate"])
    if any(float(r["input"]["relaxation_rate"]) != lam for r in records):
        raise ValueError("calibration records must share relaxation_rate")
    x = np.asarray([(float(r["input"]["temperature_a"]) - float(r["input"]["temperature_b"])) /
                    float(r["sigma"]) for r in records])
    y = np.asarray([float(r["value"]) / float(r["sigma"]) for r in records])
    conductance = float(x @ y / (x @ x))
    return float(np.clip(conductance * lam * RATE /
                         (lam * RATE - conductance * (2.0 * lam + RATE)), .4, 1.1))


class Model:
    def __init__(self): self.drag = None
    def fit(self, records):
        self.drag = fit_drag(records)
        return self
    def predict(self, experiments): return predict_at(experiments, self.drag)
