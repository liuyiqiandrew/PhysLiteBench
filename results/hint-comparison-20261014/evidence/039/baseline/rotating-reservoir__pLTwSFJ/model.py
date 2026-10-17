"""Stationary calorimeter readouts for the finite-contact rotor model."""
from functools import lru_cache

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.laguerre import laggauss
from scipy.linalg import solve
from scipy.optimize import minimize_scalar

RATE = 3.6
FIELDS = ("angular_speed", "flip_rate", "temperature_a", "temperature_b", "flux_fraction", "relaxation_rate")


def _hermites(x, degree):
    """Normalized probabilists' Hermite polynomials through ``degree``."""
    x = np.asarray(x)
    out = np.empty((x.size, degree + 1))
    out[:, 0] = 1.0
    if degree:
        out[:, 1] = x
    for n in range(1, degree):
        out[:, n + 1] = (x * out[:, n] - np.sqrt(n) * out[:, n - 1]) / np.sqrt(n + 1)
    return out


@lru_cache(maxsize=8)
def _flux_collision_unit(degree):
    """Projection of ``|v-g| (f(g,v)-f(v,g))`` at unit variance."""
    pairs = tuple((n, m) for n in range(degree + 1) for m in range(degree + 1 - n))
    q = degree + 3
    hx, hw = hermgauss(q)
    ly, lw = laggauss(q)
    centre = np.sqrt(2.0) * hx[None, :]
    matrices, weights = [], []
    for sign in (-1.0, 1.0):
        relative = sign * np.sqrt(2.0 * ly[:, None])
        v, g = (centre + relative) / np.sqrt(2.0), (centre - relative) / np.sqrt(2.0)
        hv, hg = _hermites(v.ravel(), degree), _hermites(g.ravel(), degree)
        values = np.column_stack([hv[:, n] * hg[:, m] for n, m in pairs])
        swapped = np.column_stack([hg[:, n] * hv[:, m] for n, m in pairs])
        matrices.append((values, swapped))
        weights.append((lw[:, None] * hw[None, :] / np.pi).ravel())
    before = np.concatenate([x[0] for x in matrices])
    jump = np.concatenate([x[1] - x[0] for x in matrices])
    return before.T @ (np.concatenate(weights)[:, None] * jump), pairs


def _operators(e, drag, degree=22):
    """Backward generator and derivatives of its exp(-k Q) tilt."""
    ta, tb = float(e["temperature_a"]), float(e["temperature_b"])
    lam, eta = float(e["relaxation_rate"]), float(e["flux_fraction"])
    omega, flip = float(e["angular_speed"]), float(e["flip_rate"])
    variance = max(ta, tb, 0.55)
    flux, pairs = _flux_collision_unit(degree)
    d, lookup = len(pairs), {pair: i for i, pair in enumerate(pairs)}
    base = np.zeros((d, d))
    gmul, deriv_g = np.zeros_like(base), np.zeros_like(base)
    for j, (n, m) in enumerate(pairs):
        base[j, j] -= drag * n + lam * m + RATE * (1.0 - eta)
        base[lookup[(m, n)], j] += RATE * (1.0 - eta)
        if n >= 2:
            base[lookup[(n - 2, m)], j] += drag * (ta / variance - 1.0) * np.sqrt(n * (n - 1))
        if m >= 2:
            base[lookup[(n, m - 2)], j] += lam * (tb / variance - 1.0) * np.sqrt(m * (m - 1))
        if m:
            gmul[lookup[(n, m - 1)], j] += np.sqrt(variance * m)
            deriv_g[lookup[(n, m - 1)], j] += np.sqrt(m / variance)
        if n + m < degree:
            gmul[lookup[(n, m + 1)], j] += np.sqrt(variance * (m + 1))
    base += RATE * eta * np.sqrt(variance) * flux

    # s**r h_n(v)h_m(g), r=0,1.  Since s**2=1, target drift flips parity.
    size = 2 * d
    L = np.zeros((size, size)); G = np.zeros_like(L); D = np.zeros_like(L); S = np.zeros_like(L)
    for r in range(2):
        sl, other = slice(r * d, (r + 1) * d), slice((1 - r) * d, (2 - r) * d)
        L[sl, sl] = base
        if r:
            L[sl, sl] -= 2.0 * flip * np.eye(d)
        L[other, sl] += lam * omega * deriv_g
        G[sl, sl], D[sl, sl], S[other, sl] = gmul, deriv_g, np.eye(d)
    one = np.zeros(size); one[0] = 1.0
    relative_g = G - omega * S
    identity = np.eye(size)
    # Heat and g share W_B, hence the R*d/dg term in A1.
    A1 = 2.0 * lam * tb * (relative_g @ D) - lam * (relative_g @ relative_g - tb * identity)
    A2 = 2.0 * lam * tb * (relative_g @ relative_g)
    return L, A1, A2, one


def _stationary_left(L, one):
    system, rhs = L.T.copy(), np.zeros(L.shape[0])
    i0 = int(np.flatnonzero(one)[0])
    system[i0] = 0.0; system[i0, i0] = 1.0; rhs[i0] = 1.0
    return solve(system, rhs, check_finite=False)


def _cumulants(e, drag, degree=30, detector_contrast=0.0):
    L, A1, A2, one = _operators(e, drag, degree)
    p = _stationary_left(L, one)
    i0 = int(np.flatnonzero(one)[0])
    constrained = L.copy(); constrained[i0] = p
    def inverse(rhs):
        rhs = np.asarray(rhs, dtype=float).copy(); rhs[i0] = 0.0
        return solve(constrained, rhs, check_finite=False)

    lam1 = float(p @ (A1 @ one))
    r1 = inverse(lam1 * one - A1 @ one)
    # Rapid independent phase mixing: E[w]=1 and E[w^2]=1+c^2/2.
    A2effective = (1.0 + 0.5 * float(detector_contrast) ** 2) * A2
    lam2 = float(p @ (A2effective @ one + 2.0 * A1 @ r1))
    r2 = inverse(lam2 * one + 2.0 * lam1 * r1 - 2.0 * A1 @ r1 - A2effective @ one)
    lam3 = float(p @ (3.0 * A1 @ r2 + 3.0 * A2effective @ r1))
    return {"mean": -lam1, "noise": lam2, "third_cumulant": -lam3}


def calibration_mean(e, drag):
    """Exact zero-flow, constant-clock calibration mean."""
    lam = float(e["relaxation_rate"])
    return drag * lam * RATE / (2.0 * drag * lam + RATE * (drag + lam)) * (float(e["temperature_a"]) - float(e["temperature_b"]))


def fit_drag(records):
    if not records:
        raise ValueError("at least one calibration record is required")
    def objective(drag):
        return sum(((calibration_mean(r["input"], drag) - float(r["value"])) / float(r["sigma"])) ** 2 for r in records)
    return float(minimize_scalar(objective, bounds=(0.4, 1.1), method="bounded", options={"xatol": 1e-13}).x)


@lru_cache(maxsize=4096)
def _readout_cached(values, statistic, drag, contrast):
    e = dict(zip(FIELDS, values))
    if statistic not in {"mean", "noise", "third_cumulant"}:
        raise ValueError("unknown calorimeter statistic")
    if statistic == "mean" and contrast == 0.0 and e["flux_fraction"] == 0.0 and e["angular_speed"] == 0.0:
        return calibration_mean(e, drag)
    # With only the constant collision clock the generator is quadratic and
    # low orders are already converged.  Flux collisions contain |v-g| and
    # receive the higher spectral resolution.
    degree = 8 if e["flux_fraction"] == 0.0 else 30
    return _cumulants(e, drag, degree=degree, detector_contrast=contrast)[statistic]


def predict_at(experiments, drag):
    if drag is None:
        raise RuntimeError("fit must be called before predict")
    result = []
    for experiment in experiments:
        contrast = float(experiment.get("detector_contrast", 0.0))
        if contrast and float(experiment["angular_speed"]) != 0.0:
            raise ValueError("Angularly gated preparations require zero flow target.")
        values = tuple(float(experiment[name]) for name in FIELDS)
        result.append(_readout_cached(values, experiment.get("statistic", "mean"), float(drag), contrast))
    return np.asarray(result, dtype=float)


class Model:
    def __init__(self):
        self.drag = None

    def fit(self, records):
        self.drag = fit_drag(records)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.drag)
