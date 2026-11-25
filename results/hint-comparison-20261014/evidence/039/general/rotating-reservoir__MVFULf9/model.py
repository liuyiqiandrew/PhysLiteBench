"""Stationary calorimeter predictions for the finite-contact rotor.

This is a Galerkin calculation of the backward generator.  Hermite
polynomials make the OU pieces sparse; the nonsmooth collision term is
integrated accurately in relative coordinates.  Cumulants are derivatives of
the tilted-generator eigenvalue, retaining the Brownian motion shared by the
gas and calorimeter.
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
    x = np.asarray(x)
    h = np.empty((x.size, degree + 1))
    h[:, 0] = 1.0
    if degree:
        h[:, 1] = x
    for n in range(1, degree):
        h[:, n + 1] = (x * h[:, n] - np.sqrt(n) * h[:, n - 1]) / np.sqrt(n + 1)
    return h


@lru_cache(maxsize=64)
def collision_weak_matrix(degree, variance, quadrature=None):
    """Gaussian weak matrix of |v-g|[f(g,v)-f(v,g)]."""
    variance = float(variance)
    q = int(quadrature or (2 * degree + 8))
    hx, hw = hermgauss(q)
    ly, lw = laggauss(q)
    t = np.sqrt(2.0 * variance) * hx[None, :]
    basis, swapped, weights = [], [], []
    for sign in (-1.0, 1.0):
        z = sign * np.sqrt(2.0 * variance * ly[:, None])
        v, g = (t + z) / np.sqrt(2.0), (t - z) / np.sqrt(2.0)
        hv = normalized_hermites((v / np.sqrt(variance)).ravel(), degree)
        hg = normalized_hermites((g / np.sqrt(variance)).ravel(), degree)
        basis.append((hv[:, :, None] * hg[:, None, :]).reshape(q * q, -1))
        swapped.append((hg[:, :, None] * hv[:, None, :]).reshape(q * q, -1))
        # Gaussian measure, Jacobian, and |v-g| after z^2=2*variance*y.
        weights.append((np.sqrt(variance) / np.pi * lw[:, None] * hw[None, :]).ravel())
    b = np.concatenate(basis)
    return b.T @ (np.concatenate(weights)[:, None] * (np.concatenate(swapped) - b))


def _operators(e, drag, degree):
    """Backward generator and first two exp(k Q) tilt derivatives."""
    ta, tb = float(e["temperature_a"]), float(e["temperature_b"])
    omega, relaxation = float(e["angular_speed"]), float(e["relaxation_rate"])
    eta, flip = float(e["flux_fraction"]), float(e["flip_rate"])
    variance = max(ta, tb, 0.35 * omega * omega + 0.5)
    width, nbase = degree + 1, (degree + 1) ** 2
    index = lambda n, m: n * width + m

    base = np.zeros((nbase, nbase))
    gmul, dg, g2 = np.zeros_like(base), np.zeros_like(base), np.zeros_like(base)
    eye = np.eye(nbase)
    for n in range(width):
        for m in range(width):
            j = index(n, m)
            base[j, j] -= drag * n + relaxation * m
            if n >= 2:
                base[index(n - 2, m), j] += drag * (ta / variance - 1.0) * np.sqrt(n * (n - 1))
            if m >= 2:
                base[index(n, m - 2), j] += relaxation * (tb / variance - 1.0) * np.sqrt(m * (m - 1))
            if m:
                dg[index(n, m - 1), j] = np.sqrt(m / variance)
                gmul[index(n, m - 1), j] = np.sqrt(variance * m)
            if m < degree:
                gmul[index(n, m + 1), j] = np.sqrt(variance * (m + 1))
            g2[j, j] = variance * (2 * m + 1)
            if m >= 2:
                g2[index(n, m - 2), j] = variance * np.sqrt(m * (m - 1))
            if m + 2 <= degree:
                g2[index(n, m + 2), j] = variance * np.sqrt((m + 1) * (m + 2))

    swap = np.zeros_like(base)
    for n in range(width):
        for m in range(width):
            swap[index(m, n), index(n, m)] = 1.0
    base += RATE * (1.0 - eta) * (swap - eye)
    if eta:
        base += RATE * eta * collision_weak_matrix(degree, variance)

    size = 2 * nbase
    L, A1, A2 = np.zeros((size, size)), np.zeros((size, size)), np.zeros((size, size))
    for block, sign in enumerate((1.0, -1.0)):
        sl = slice(block * nbase, (block + 1) * nbase)
        L[sl, sl] = base + relaxation * sign * omega * dg
        x = gmul - sign * omega * eye
        x2 = g2 - 2.0 * sign * omega * gmul + omega * omega * eye
        # a=lambda(x^2-T), b=-sqrt(2lambdaT)x; b*sigma*d_g is the shared-noise term.
        A1[sl, sl] = relaxation * (x2 - tb * eye) - 2.0 * relaxation * tb * (x @ dg)
        A2[sl, sl] = 2.0 * relaxation * tb * x2
    for j in range(nbase):
        L[j, j] -= flip
        L[nbase + j, nbase + j] -= flip
        L[nbase + j, j] += flip
        L[j, nbase + j] += flip
    constant = np.zeros(size)
    constant[0] = constant[nbase] = 1.0
    return L, A1, A2, constant


def _statistics(e, drag, degree, variance_factor=1.0):
    L, A1, A2, constant = _operators(e, drag, degree)
    A2 *= variance_factor
    # Invariant expectation functional p: p L=0 and p 1=1.
    forward = L.T.copy()
    forward[0, :] = constant
    target = np.zeros(len(constant))
    target[0] = 1.0
    p = solve(forward, target, check_finite=False)

    def poisson(rhs):
        system = L.copy()
        system[0, :] = p
        rhs = np.asarray(rhs).copy()
        rhs[0] = 0.0
        return solve(system, rhs, check_finite=False)

    c1 = float(p @ (A1 @ constant))
    r1 = poisson(c1 * constant - A1 @ constant)
    c2 = float(p @ (A2 @ constant + 2.0 * A1 @ r1))
    r2 = poisson(c2 * constant + 2.0 * c1 * r1 - 2.0 * A1 @ r1 - A2 @ constant)
    c3 = float(p @ (3.0 * A1 @ r2 + 3.0 * A2 @ r1))
    return {"mean": c1, "noise": c2, "third_cumulant": c3}


def calibration_mean(e, drag):
    relaxation = float(e["relaxation_rate"])
    conductance = drag * relaxation * RATE / (2.0 * drag * relaxation + RATE * (drag + relaxation))
    return conductance * (float(e["temperature_a"]) - float(e["temperature_b"]))


def _degree(e):
    # For eta=0 every response needed here is a finite Gaussian polynomial.
    # Flux collisions have a cusp and need a larger basis.
    return 6 if float(e["flux_fraction"]) == 0.0 else 28


@lru_cache(maxsize=2048)
def _ungated_all(angular_speed, flip_rate, ta, tb, eta, relaxation, drag):
    e = dict(zip(FIELDS, (angular_speed, flip_rate, ta, tb, eta, relaxation)))
    return _statistics(e, drag, _degree(e))


@lru_cache(maxsize=2048)
def _gated_all(ta, tb, eta, relaxation, contrast, drag):
    """The documented independently rapidly mixed detector-phase limit."""
    # E[w]=1 and E[w^2]=1+contrast^2/2.  Thus only the local quadratic tilt
    # changes; detector harmonic and motor flips are irrelevant at zero flow.
    e = dict(angular_speed=0.0, flip_rate=1.0, temperature_a=ta, temperature_b=tb,
             flux_fraction=eta, relaxation_rate=relaxation)
    return _statistics(e, drag, _degree(e), 1.0 + contrast * contrast / 2.0)


def predict_at(experiments, drag):
    if drag is None:
        raise RuntimeError("fit must be called before predict")
    result = []
    for e in experiments:
        contrast, statistic = float(e.get("detector_contrast", 0.0)), e.get("statistic", "mean")
        if statistic not in ("mean", "noise", "third_cumulant"):
            raise ValueError("unknown calorimeter statistic")
        if contrast:
            if float(e["angular_speed"]) != 0.0:
                raise ValueError("Angularly gated preparations require zero flow target.")
            values = _gated_all(float(e["temperature_a"]), float(e["temperature_b"]),
                                float(e["flux_fraction"]), float(e["relaxation_rate"]),
                                contrast, float(drag))
        else:
            if statistic == "mean" and float(e["flux_fraction"]) == 0.0 and float(e["angular_speed"]) == 0.0:
                result.append(calibration_mean(e, float(drag)))
                continue
            values = _ungated_all(*(float(e[f]) for f in FIELDS), float(drag))
        result.append(values[statistic])
    return np.asarray(result, dtype=float)


class Model:
    def __init__(self):
        self.drag = None

    def fit(self, records):
        if not records:
            raise ValueError("at least one calibration record is required")
        rates = {float(r["input"]["relaxation_rate"]) for r in records}
        if len(rates) != 1:
            raise ValueError("calibration records must share relaxation_rate")
        relaxation = rates.pop()
        delta = np.asarray([float(r["input"]["temperature_a"]) - float(r["input"]["temperature_b"])
                            for r in records])
        sigma = np.asarray([float(r["sigma"]) for r in records])
        values = np.asarray([float(r["value"]) for r in records])
        weights = 1.0 / sigma**2
        denom = float(np.sum(weights * delta**2))
        if denom == 0.0:
            raise ValueError("calibration temperatures contain no contrast")
        conductance = float(np.sum(weights * delta * values) / denom)
        drag = conductance * relaxation * RATE / (relaxation * RATE - conductance * (2.0 * relaxation + RATE))
        self.drag = float(np.clip(drag, 0.4, 1.1))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.drag)
