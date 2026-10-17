"""Stationary calorimeter predictions for the finite-contact rotor model.

The calculation is a Galerkin calculation for the backward generator.
Hermite polynomials make the OU parts sparse, while a Laguerre rule represents
the physical absolute-relative-speed collision clock without smoothing it.
"""
from functools import lru_cache

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.laguerre import laggauss
from scipy.linalg import solve

RATE = 3.6
FIELDS = ("angular_speed", "flip_rate", "temperature_a", "temperature_b",
          "flux_fraction", "relaxation_rate")


def normalized_hermites(x, degree):
    """Probabilists' Hermites normalized for a standard normal law."""
    x = np.asarray(x)
    out = np.empty((x.size, degree + 1))
    out[:, 0] = 1.0
    if degree:
        out[:, 1] = x
    for n in range(1, degree):
        out[:, n + 1] = (x * out[:, n] - np.sqrt(n) * out[:, n - 1]) / np.sqrt(n + 1)
    return out


@lru_cache(maxsize=128)
def collision_weak_matrix(degree, variance):
    """Projection of ``|v-g| (f(g,v)-f(v,g))`` in tensor Hermites."""
    q = 2 * degree + 4
    hx, hw = hermgauss(q)
    ly, lw = laggauss(q)
    centre = np.sqrt(2.0 * variance) * hx[None, :]
    blocks, exchanged, weights = [], [], []
    for sign in (-1.0, 1.0):
        relative = sign * np.sqrt(2.0 * variance * ly[:, None])
        v = (centre + relative) / np.sqrt(2.0)
        g = (centre - relative) / np.sqrt(2.0)
        hv = normalized_hermites(v.ravel() / np.sqrt(variance), degree)
        hg = normalized_hermites(g.ravel() / np.sqrt(variance), degree)
        base = (hv[:, :, None] * hg[:, None, :]).reshape(q * q, -1)
        other = (hg[:, :, None] * hv[:, None, :]).reshape(q * q, -1)
        blocks.append(base)
        exchanged.append(other)
        # This includes the collision relative speed sqrt(2)*|relative|.
        weights.append((np.sqrt(variance) / np.pi * lw[:, None] * hw[None, :]).ravel())
    base = np.concatenate(blocks)
    jump = np.concatenate(exchanged) - base
    return base.T @ (np.concatenate(weights)[:, None] * jump)


@lru_cache(maxsize=32)
def _indices(degree):
    return tuple((n, m) for n in range(degree + 1) for m in range(degree + 1 - n))


def _system(e, drag, degree):
    """Build generator and Feynman--Kac heat derivatives D1/D2.

    The two blocks represent the motor-label polynomial basis 1 and s.  L
    acts on observable coefficients; p is the normalized left null vector.
    """
    ta, tb = float(e["temperature_a"]), float(e["temperature_b"])
    omega, lam = float(e["angular_speed"]), float(e["relaxation_rate"])
    eta, flip, drag = float(e["flux_fraction"]), float(e["flip_rate"]), float(drag)
    variance = max(ta, tb, 0.35 + omega * omega)
    pairs = _indices(degree)
    npairs = len(pairs)
    pos = {pair: i for i, pair in enumerate(pairs)}
    size = 2 * npairs
    ix = lambda parity, n, m: parity * npairs + pos[(n, m)]
    L = np.zeros((size, size))
    G = np.zeros_like(L)     # multiplication by g
    DG = np.zeros_like(L)    # d/dg
    G2 = np.zeros_like(L)    # multiplication by g squared
    S = np.zeros_like(L)     # multiplication by s

    for parity in range(2):
        for n, m in pairs:
            j = ix(parity, n, m)
            L[j, j] -= drag * n + lam * m + (2.0 * flip if parity else 0.0)
            if n >= 2:
                L[ix(parity, n - 2, m), j] += drag * (ta / variance - 1.0) * np.sqrt(n * (n - 1))
            if m >= 2:
                L[ix(parity, n, m - 2), j] += lam * (tb / variance - 1.0) * np.sqrt(m * (m - 1))
            # +lambda*s*omega*d_g toggles motor-label parity.
            if m:
                L[ix(1 - parity, n, m - 1), j] += lam * omega * np.sqrt(m / variance)

            if m:
                G[ix(parity, n, m - 1), j] += np.sqrt(variance * m)
                DG[ix(parity, n, m - 1), j] += np.sqrt(m / variance)
            if (n, m + 1) in pos:
                G[ix(parity, n, m + 1), j] += np.sqrt(variance * (m + 1))
            G2[j, j] += variance * (2 * m + 1)
            if m >= 2:
                G2[ix(parity, n, m - 2), j] += variance * np.sqrt(m * (m - 1))
            if (n, m + 2) in pos:
                G2[ix(parity, n, m + 2), j] += variance * np.sqrt((m + 1) * (m + 2))
            S[ix(1 - parity, n, m), j] = 1.0

    # The constant-rate collision channel is exact at every retained degree.
    for parity in range(2):
        for n, m in pairs:
            j = ix(parity, n, m)
            L[j, j] -= RATE * (1.0 - eta)
            L[ix(parity, m, n), j] += RATE * (1.0 - eta)
    if eta:
        full = collision_weak_matrix(degree, float(variance))
        full_index = lambda n, m: n * (degree + 1) + m
        selected = [full_index(n, m) for n, m in pairs]
        collision = full[np.ix_(selected, selected)]
        L[:npairs, :npairs] += RATE * eta * collision
        L[npairs:, npairs:] += RATE * eta * collision

    one = np.zeros(size)
    one[ix(0, 0, 0)] = 1.0
    identity = np.eye(size)
    # dQ=a dt+b dWg.  b and dg share Wg, producing b*sigma_g*d_g in D1.
    relative = G - omega * S
    relative2 = G2 - 2.0 * omega * (G @ S) + omega * omega * identity
    D1 = lam * (relative2 - tb * identity) - 2.0 * lam * tb * (relative @ DG)
    D2 = 2.0 * lam * tb * relative2

    stationary = L.T.copy()
    stationary[0, :] = 0.0
    stationary[0, 0] = 1.0
    rhs = np.zeros(size)
    rhs[0] = 1.0
    p = solve(stationary, rhs, check_finite=False)
    return L, D1, D2, p, one


def _rates(e, drag, gate_square=1.0):
    # At zero flux the finite polynomial hierarchy closes through degree six.
    degree = 6 if float(e["flux_fraction"]) == 0.0 else 32
    L, D1, D2, p, one = _system(e, drag, degree)
    D2 = gate_square * D2
    c1 = float(p @ (D1 @ one))
    linear = L.copy()
    linear[0, :] = p
    rhs = c1 * one - D1 @ one
    rhs[0] = 0.0
    r1 = solve(linear, rhs, check_finite=False)
    c2 = float(p @ (D2 @ one + 2.0 * D1 @ r1))
    rhs = c2 * one + 2.0 * c1 * r1 - 2.0 * D1 @ r1 - D2 @ one
    rhs[0] = 0.0
    r2 = solve(linear, rhs, check_finite=False)
    c3 = float(p @ (3.0 * D1 @ r2 + 3.0 * D2 @ r1))
    return {"mean": c1, "noise": c2, "third_cumulant": c3}


@lru_cache(maxsize=2048)
def _cached(values, contrast, statistic, drag):
    e = dict(zip(FIELDS, values))
    # Documented gated approximation: phase is independently rapidly mixed.
    # Hence <w>=1 and <w^2>=1+contrast^2/2; harmonic drops out.
    rates = _rates(e, drag, 1.0 + contrast * contrast / 2.0)
    return rates[statistic]


def calibration_mean(e, drag):
    """Exact calibration conductance (zero flow and zero flux)."""
    lam = float(e["relaxation_rate"])
    d = float(drag)
    return d * lam * RATE / (2.0 * d * lam + RATE * (d + lam)) * (
        float(e["temperature_a"]) - float(e["temperature_b"]))


def fit_drag(records):
    if not records:
        raise ValueError("at least one calibration record is required")
    lam = float(records[0]["input"]["relaxation_rate"])
    if any(float(row["input"]["relaxation_rate"]) != lam for row in records):
        raise ValueError("calibration records must share relaxation_rate")
    x = np.asarray([(float(row["input"]["temperature_a"]) - float(row["input"]["temperature_b"])) /
                    float(row["sigma"]) for row in records])
    y = np.asarray([float(row["value"]) / float(row["sigma"]) for row in records])
    conductance = float(x @ y / (x @ x))
    denominator = lam * RATE - conductance * (2.0 * lam + RATE)
    return float(np.clip(conductance * lam * RATE / denominator, 0.4, 1.1))


def predict_at(experiments, drag):
    if drag is None:
        raise RuntimeError("fit must be called before predict")
    result = []
    for e in experiments:
        statistic = e.get("statistic", "mean")
        if statistic not in {"mean", "noise", "third_cumulant"}:
            raise ValueError("unknown calorimeter statistic")
        values = tuple(float(e[name]) for name in FIELDS)
        contrast = float(e.get("detector_contrast", 0.0))
        if contrast and values[0] != 0.0:
            raise ValueError("Angularly gated preparations require zero flow target")
        # Exact formula is faster and avoids gratuitous numerical error on all
        # calibration-form means (detector contrast is necessarily zero there).
        if not contrast and statistic == "mean" and values[0] == 0.0 and values[4] == 0.0:
            value = calibration_mean(e, drag)
        else:
            value = _cached(values, contrast, statistic, float(drag))
        result.append(value)
    result = np.asarray(result, dtype=float)
    if not np.isfinite(result).all():
        raise FloatingPointError("non-finite stationary prediction")
    return result


class Model:
    def __init__(self):
        self.drag = None

    def fit(self, records):
        self.drag = fit_drag(records)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.drag)
