"""Stationary moment model for the finite rotating-gas contact.

The implementation uses a Hermite-Galerkin representation of the backward
generator.  The motor state is kept explicitly (as a two-state telegraph
process), and heat statistics come from the tilted generator for the same
Wiener process that drives the gas velocity.
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
STATISTICS = ("mean", "noise", "third_cumulant")


def normalized_hermites(z, degree):
    z = np.asarray(z)
    h = np.zeros((z.size, degree + 1))
    h[:, 0] = 1.0
    if degree:
        h[:, 1] = z
    for n in range(1, degree):
        h[:, n + 1] = (z * h[:, n] - np.sqrt(n) * h[:, n - 1]) / np.sqrt(n + 1)
    return h


@lru_cache(maxsize=32)
def collision_weak_matrix(degree, variance=1.6, quadrature=None):
    """Galerkin matrix of abs(v-g)*(swap-identity)."""
    degree = int(degree)
    variance = float(variance)
    q = int(quadrature or (2 * degree + 4))
    th, tw = hermgauss(q)
    yl, yw = laggauss(q)
    t = np.sqrt(2.0 * variance) * th[None, :]
    blocks, swapped, weights = [], [], []
    for sign in (-1.0, 1.0):
        z = sign * np.sqrt(2.0 * variance * yl[:, None])
        v = (t + z) / np.sqrt(2.0)
        g = (t - z) / np.sqrt(2.0)
        hv = normalized_hermites(v.ravel() / np.sqrt(variance), degree)
        hg = normalized_hermites(g.ravel() / np.sqrt(variance), degree)
        blocks.append((hv[:, :, None] * hg[:, None, :]).reshape(q * q, -1))
        swapped.append((hg[:, :, None] * hv[:, None, :]).reshape(q * q, -1))
        weights.append((np.sqrt(variance) / np.pi * yw[:, None] * tw[None, :]).ravel())
    b = np.concatenate(blocks)
    result = b.T @ (np.concatenate(weights)[:, None] * (np.concatenate(swapped) - b))
    if np.max(np.abs(result - result.T)) > 2e-8:
        raise ArithmeticError("collision quadrature lost self-adjointness")
    return result


def _scalar_operators(e, drag, degree):
    """Operators for v,g with zero flow target and no motor state."""
    ta, tb = float(e["temperature_a"]), float(e["temperature_b"])
    lam, eta = float(e["relaxation_rate"]), float(e["flux_fraction"])
    variance = max(ta, tb)
    width, size = degree + 1, (degree + 1) ** 2
    index = lambda n, m: n * width + m
    L = np.zeros((size, size))
    V, G, D, G2 = (np.zeros_like(L) for _ in range(4))
    for n in range(width):
        for m in range(width):
            j = index(n, m)
            L[j, j] -= drag * n + lam * m + RATE * (1.0 - eta)
            L[index(m, n), j] += RATE * (1.0 - eta)
            if n >= 2:
                L[index(n - 2, m), j] += drag * (ta / variance - 1.0) * np.sqrt(n * (n - 1))
            if m >= 2:
                L[index(n, m - 2), j] += lam * (tb / variance - 1.0) * np.sqrt(m * (m - 1))
            if n >= 1:
                V[index(n - 1, m), j] = np.sqrt(variance * n)
            if n + 1 < width:
                V[index(n + 1, m), j] = np.sqrt(variance * (n + 1))
            if m >= 1:
                G[index(n, m - 1), j] = np.sqrt(variance * m)
                D[index(n, m - 1), j] = np.sqrt(m / variance)
            if m + 1 < width:
                G[index(n, m + 1), j] = np.sqrt(variance * (m + 1))
            G2[j, j] = variance * (2.0 * m + 1.0)
            if m >= 2:
                G2[index(n, m - 2), j] = variance * np.sqrt(m * (m - 1))
            if m + 2 < width:
                G2[index(n, m + 2), j] = variance * np.sqrt((m + 1) * (m + 2))
    L += RATE * eta * collision_weak_matrix(degree, variance)
    return L, V, G, D, G2, variance


def _stationary(L, constant_index=0):
    forward = L.T.copy()
    forward[constant_index] = 0.0
    forward[constant_index, constant_index] = 1.0
    rhs = np.zeros(L.shape[0])
    rhs[constant_index] = 1.0
    return solve(forward, rhs, check_finite=False)


def _tilted_statistics(L, A1, A2, p, constant_index=0):
    """Cumulants of Q from the eigenvalue of E exp(-k Q)."""
    constant = np.zeros(L.shape[0])
    constant[constant_index] = 1.0
    one_a1, one_a2 = A1[:, constant_index], A2[:, constant_index]
    mu1 = float(p @ one_a1)
    system = L.copy()
    system[constant_index] = p
    rhs1 = mu1 * constant - one_a1
    rhs1[constant_index] = 0.0
    r1 = solve(system, rhs1, check_finite=False)
    mu2 = float(2.0 * (p @ A1 @ r1) + p @ one_a2)
    rhs2 = mu2 * constant + 2.0 * mu1 * r1 - 2.0 * (A1 @ r1) - one_a2
    rhs2[constant_index] = 0.0
    r2 = solve(system, rhs2, check_finite=False)
    mu3 = float(p @ (3.0 * (A1 @ r2) + 3.0 * (A2 @ r1)))
    return {"mean": -mu1, "noise": mu2, "third_cumulant": -mu3}


def _motor_operators(e, drag, degree):
    """Full generator in the basis h_n(v) h_m(g) s^k, k=0,1."""
    tb = float(e["temperature_b"])
    omega = float(e["angular_speed"])
    lam, flip = float(e["relaxation_rate"]), float(e["flip_rate"])
    scalar = dict(e)
    scalar["angular_speed"] = 0.0
    L0, _V0, G0, D0, G20, _variance = _scalar_operators(scalar, drag, degree)
    d, size = L0.shape[0], 2 * L0.shape[0]
    L = np.zeros((size, size))
    L[:d, :d], L[d:, d:] = L0, L0 - 2.0 * flip * np.eye(d)
    L[d:, :d] += lam * omega * D0
    L[:d, d:] += lam * omega * D0

    def lift(op):
        out = np.zeros((size, size))
        out[:d, :d], out[d:, d:] = op, op
        return out

    S = np.zeros((size, size))
    S[:d, d:], S[d:, :d] = np.eye(d), np.eye(d)
    G, D, G2 = lift(G0), lift(D0), lift(G20)
    X2 = G2 - 2.0 * omega * (G @ S) + omega * omega * np.eye(size)
    XD = G @ D - omega * (S @ D)
    A1 = -lam * (X2 - tb * np.eye(size)) + 2.0 * lam * tb * XD
    A2 = 2.0 * lam * tb * X2
    return L, A1, A2, _stationary(L)


def calibration_mean(e, drag):
    lam = float(e["relaxation_rate"])
    return (drag * lam * RATE / (2.0 * drag * lam + RATE * (drag + lam))
            * (float(e["temperature_a"]) - float(e["temperature_b"])))


def kinetic_statistics(e, drag, degree=22, variance=None):
    if float(e["flux_fraction"]) == 0.0 and float(e["angular_speed"]) == 0.0:
        result = {"mean": float(calibration_mean(e, drag))}
    else:
        result = {}
    L, A1, A2, p = _motor_operators(e, drag, degree)
    result.update(_tilted_statistics(L, A1, A2, p))
    return result


def angular_hermite(e, gamma=0.67, degree=22, averaged=False, phase_mixing_rate=0.0):
    """Cumulants under the documented rapidly mixed detector-phase limit."""
    if float(e.get("angular_speed", 0.0)) != 0.0:
        raise ValueError("angularly gated preparations require zero flow target")
    eps = float(e.get("detector_contrast", 0.0))
    L, _V, G, D, G2, _variance = _scalar_operators(dict(e), gamma, degree)
    lam, tb = float(e["relaxation_rate"]), float(e["temperature_b"])
    A1 = -lam * (G2 - tb * np.eye(L.shape[0])) + 2.0 * lam * tb * (G @ D)
    A2 = 2.0 * lam * tb * G2 * (1.0 + 0.5 * eps * eps)
    return _tilted_statistics(L, A1, A2, _stationary(L))


def fit_drag(records):
    if not records:
        raise ValueError("at least one calibration record is required")
    lam = float(records[0]["input"]["relaxation_rate"])
    x, y = [], []
    for record in records:
        inp, sigma = record["input"], float(record["sigma"])
        if float(inp["angular_speed"]) != 0.0 or float(inp["flux_fraction"]) != 0.0:
            raise ValueError("calibration records do not match the documented preparation")
        if not np.isclose(float(inp["relaxation_rate"]), lam):
            raise ValueError("calibration relaxation rates must agree")
        if sigma <= 0.0:
            raise ValueError("calibration uncertainties must be positive")
        x.append((float(inp["temperature_a"]) - float(inp["temperature_b"])) / sigma)
        y.append(float(record["value"]) / sigma)
    x, y = np.asarray(x), np.asarray(y)
    if not np.any(x):
        raise ValueError("calibration does not identify a conductance")
    conductance = float(x @ y / (x @ x))
    denominator = lam * RATE - conductance * (2.0 * lam + RATE)
    if denominator == 0.0:
        raise ValueError("calibration fit is incompatible with a finite drag")
    return float(np.clip(conductance * lam * RATE / denominator, 0.4, 1.1))


@lru_cache(maxsize=1024)
def readout(angular_speed, flip_rate, temperature_a, temperature_b,
            flux_fraction, relaxation_rate, statistic, drag):
    if statistic not in STATISTICS:
        raise ValueError("unknown calorimeter statistic")
    e = dict(zip(FIELDS, (angular_speed, flip_rate, temperature_a, temperature_b,
                          flux_fraction, relaxation_rate)))
    if statistic == "mean" and flux_fraction == 0.0 and angular_speed == 0.0:
        return float(calibration_mean(e, drag))
    return float(kinetic_statistics(e, drag)[statistic])


@lru_cache(maxsize=512)
def angular_readouts(temperature_a, temperature_b, relaxation_rate,
                     flux_fraction, contrast, harmonic, drag):
    e = dict(angular_speed=0.0, temperature_a=temperature_a, temperature_b=temperature_b,
             relaxation_rate=relaxation_rate, flux_fraction=flux_fraction,
             detector_contrast=contrast, detector_harmonic=harmonic)
    return angular_hermite(e, drag)


def predict_at(experiments, drag):
    result = []
    for e in experiments:
        statistic = e.get("statistic", "mean")
        if statistic not in STATISTICS:
            raise ValueError("unknown calorimeter statistic")
        contrast = float(e.get("detector_contrast", 0.0))
        if contrast != 0.0:
            if float(e["angular_speed"]) != 0.0:
                raise ValueError("angularly gated preparations require zero flow target")
            values = angular_readouts(float(e["temperature_a"]), float(e["temperature_b"]),
                                      float(e["relaxation_rate"]), float(e["flux_fraction"]),
                                      contrast, int(e.get("detector_harmonic", 1)), float(drag))
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
        if self.drag is None:
            raise RuntimeError("Model.fit must be called before predict")
        return predict_at(experiments, self.drag)
