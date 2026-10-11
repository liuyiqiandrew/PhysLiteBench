"""Stationary calorimeter statistics for the finite retained gas contact.

This is a Hermite--Galerkin solution of the backward generator.  The two OU
generators and the heat tilt are sparse; the non-polynomial collision term is
integrated accurately by Gaussian quadrature.  The motor is retained as its
two actual states, since replacing it by its mean flow loses heat noise.
"""
from functools import lru_cache

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.laguerre import laggauss
from scipy.linalg import solve
from scipy.optimize import minimize_scalar

RATE = 3.6
FIELDS = ("angular_speed", "flip_rate", "temperature_a", "temperature_b",
          "flux_fraction", "relaxation_rate")


def _hermites(x, degree):
    """Orthonormal probabilists' Hermite polynomials at x."""
    x = np.asarray(x)
    ans = np.empty((x.size, degree + 1)); ans[:, 0] = 1.
    if degree: ans[:, 1] = x
    for n in range(1, degree):
        ans[:, n + 1] = (x * ans[:, n] - np.sqrt(n) * ans[:, n - 1]) / np.sqrt(n + 1)
    return ans


@lru_cache(maxsize=64)
def _collision_matrix(degree, variance):
    """Matrix of |v-g| (f(g,v)-f(v,g)) in a Gaussian Hermite product basis."""
    # z=(v-g)/sqrt(2), t=(v+g)/sqrt(2).  Laguerre deals with |z|.
    q = 2 * degree + 8
    hx, hw = hermgauss(q); ly, lw = laggauss(q)
    t = np.sqrt(2. * variance) * hx[None, :]
    pieces, swapped, weights = [], [], []
    for sign in (-1., 1.):
        z = sign * np.sqrt(2. * variance * ly[:, None])
        v, g = (t + z) / np.sqrt(2.), (t - z) / np.sqrt(2.)
        pv = _hermites((v / np.sqrt(variance)).ravel(), degree)
        pg = _hermites((g / np.sqrt(variance)).ravel(), degree)
        pieces.append((pv[:, :, None] * pg[:, None, :]).reshape(q * q, -1))
        swapped.append((pg[:, :, None] * pv[:, None, :]).reshape(q * q, -1))
        weights.append((np.sqrt(variance) / np.pi * lw[:, None] * hw[None, :]).ravel())
    basis = np.concatenate(pieces)
    out = basis.T @ (np.concatenate(weights)[:, None] * (np.concatenate(swapped) - basis))
    if not np.allclose(out, out.T, rtol=0, atol=2.e-10):
        raise ArithmeticError("collision quadrature lost swap symmetry")
    return out


def _basis(degree):
    pairs = [(n, m) for n in range(degree + 1) for m in range(degree + 1 - n)]
    return pairs, {pair: i for i, pair in enumerate(pairs)}


def _operators(e, drag, degree=24):
    """Return L, heat-tilt derivatives A1/A2, stationary p, and constant one.

    L acts on backward test functions.  A1/A2 correspond to an exp(-k Q)
    tilt, hence physical odd cumulants have the opposite eigenvalue sign.
    """
    ta, tb = float(e["temperature_a"]), float(e["temperature_b"])
    omega, lam = float(e["angular_speed"]), float(e["relaxation_rate"])
    eta, flip = float(e["flux_fraction"]), float(e["flip_rate"])
    variance = max(ta, tb, .4)
    pairs, ix = _basis(degree); nb = len(pairs); size = 2 * nb
    L = np.zeros((size, size)); A1 = np.zeros_like(L); A2 = np.zeros_like(L)

    def ind(s, n, m):
        k = ix.get((n, m))
        return None if k is None else s * nb + k
    def put(matrix, row, col, value):
        if row is not None: matrix[row, col] += value

    for s in range(2):
        u = (2 * s - 1) * omega
        for n, m in pairs:
            j = ind(s, n, m)
            # OU generators, motor switching, and constant-rate collisions.
            L[j, j] -= drag * n + lam * m + flip + RATE * (1. - eta)
            put(L, ind(1-s, n, m), j, flip)
            put(L, ind(s, m, n), j, RATE * (1. - eta))
            if n >= 2: put(L, ind(s, n-2, m), j, drag * (ta/variance-1.) * np.sqrt(n*(n-1)))
            if m >= 2: put(L, ind(s, n, m-2), j, lam * (tb/variance-1.) * np.sqrt(m*(m-1)))
            if m >= 1: put(L, ind(s, n, m-1), j, lam*u/np.sqrt(variance) * np.sqrt(m))

            # A1=-a-h*sigma*d_g, A2=h^2 for dQ=a dt+h dW.
            # -lambda[(g-u)^2-T] contribution
            A1[j, j] += -lam * (variance*(2*m+1) + u*u - tb)
            if m >= 2: put(A1, ind(s, n, m-2), j, -lam*variance*np.sqrt(m*(m-1)))
            put(A1, ind(s, n, m+2), j, -lam*variance*np.sqrt((m+1)*(m+2)))
            if m >= 1: put(A1, ind(s, n, m-1), j, 2.*lam*u*np.sqrt(variance*m))
            put(A1, ind(s, n, m+1), j, 2.*lam*u*np.sqrt(variance*(m+1)))
            # +2 lambda T (g-u) d/dg contribution
            A1[j, j] += 2.*lam*tb*m
            if m >= 2: put(A1, ind(s, n, m-2), j, 2.*lam*tb*np.sqrt(m*(m-1)))
            if m >= 1: put(A1, ind(s, n, m-1), j, -2.*lam*tb*u*np.sqrt(m/variance))

            A2[j, j] += 2.*lam*tb * (variance*(2*m+1) + u*u)
            if m >= 2: put(A2, ind(s, n, m-2), j, 2.*lam*tb*variance*np.sqrt(m*(m-1)))
            put(A2, ind(s, n, m+2), j, 2.*lam*tb*variance*np.sqrt((m+1)*(m+2)))
            if m >= 1: put(A2, ind(s, n, m-1), j, -4.*lam*tb*u*np.sqrt(variance*m))
            put(A2, ind(s, n, m+1), j, -4.*lam*tb*u*np.sqrt(variance*(m+1)))

    full = _collision_matrix(degree, variance)
    full_index = lambda n, m: n * (degree + 1) + m
    for s in range(2):
        off = s * nb
        for i, (ni, mi) in enumerate(pairs):
            fi = full_index(ni, mi)
            for j, (nj, mj) in enumerate(pairs):
                L[off+i, off+j] += RATE * eta * full[fi, full_index(nj, mj)]

    one = np.zeros(size); one[ind(0, 0, 0)] = one[ind(1, 0, 0)] = 1.
    system = L.T.copy(); system[0] = one
    rhs = np.zeros(size); rhs[0] = 1.
    p = solve(system, rhs, check_finite=False)
    return L, A1, A2, p, one


def _statistics(e, drag, contrast=0.):
    """Long-run mean, noise and third-cumulant rates.

    The documented gated predictor uses the independently rapid phase-mixing
    limit: E[w]=1 and E[w^2]=1+contrast^2/2.
    """
    L, A1, A2, p, one = _operators(e, drag)
    alpha = 1. + .5 * contrast * contrast
    l1 = float(p @ (A1 @ one))
    constrained = L.copy(); constrained[0] = p
    rhs1 = l1*one - A1@one; rhs1[0] = 0.
    r1 = solve(constrained, rhs1, check_finite=False)
    l2 = float(p @ (alpha*(A2@one) + 2.*A1@r1))
    rhs2 = l2*one + 2.*l1*r1 - 2.*A1@r1 - alpha*(A2@one); rhs2[0] = 0.
    r2 = solve(constrained, rhs2, check_finite=False)
    l3 = float(p @ (3.*A1@r2 + 3.*alpha*A2@r1))
    return {"mean": -l1, "noise": l2, "third_cumulant": -l3}


def _conductance(drag, lam):
    return drag * lam * RATE / (2.*drag*lam + RATE*(drag + lam))


def fit_drag(records):
    """Weighted one-parameter fit to the exact zero-flow calibration law."""
    if not records: raise ValueError("at least one calibration record is required")
    x = np.asarray([r["input"]["temperature_a"]-r["input"]["temperature_b"] for r in records])
    y = np.asarray([r["value"] for r in records]); sigma = np.asarray([r["sigma"] for r in records])
    lam = np.asarray([r["input"]["relaxation_rate"] for r in records])
    def objective(d):
        residual = (_conductance(d, lam)*x-y)/sigma
        return float(residual@residual)
    return float(minimize_scalar(objective, bounds=(.4, 1.1), method="bounded", options={"xatol":1e-12}).x)


@lru_cache(maxsize=4096)
def _readout(angular_speed, flip_rate, temperature_a, temperature_b, flux_fraction,
             relaxation_rate, statistic, contrast, drag):
    if statistic not in ("mean", "noise", "third_cumulant"): raise ValueError("unknown calorimeter statistic")
    if statistic == "mean" and contrast == 0. and flux_fraction == 0. and angular_speed == 0.:
        return _conductance(drag, relaxation_rate) * (temperature_a-temperature_b)
    e = dict(zip(FIELDS, (angular_speed, flip_rate, temperature_a, temperature_b, flux_fraction, relaxation_rate)))
    return _statistics(e, drag, contrast)[statistic]


def predict_at(experiments, drag):
    if drag is None: raise RuntimeError("fit must be called before predict")
    values = []
    for e in experiments:
        contrast, speed = float(e.get("detector_contrast", 0.)), float(e["angular_speed"])
        if contrast and speed != 0.: raise ValueError("Angularly gated preparations require zero flow target.")
        values.append(_readout(*(float(e[name]) for name in FIELDS), e.get("statistic", "mean"), contrast, float(drag)))
    return np.asarray(values, dtype=float)


class Model:
    def __init__(self): self.drag = None
    def fit(self, records):
        self.drag = fit_drag(records)
        return self
    def predict(self, experiments): return predict_at(experiments, self.drag)
