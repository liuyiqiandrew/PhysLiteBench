"""Stationary calorimeter statistics for the finite-contact rotor model.

The implementation uses a Galerkin representation of the backward generator.
Hermite modes describe ``v`` and ``g`` and a two-state mode describes the
motor label.  For a gated detector, Fourier modes in the physical angle are
kept; a uniform initial angle is therefore not treated as a sequence of
independent angles.
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
STATISTICS = ("mean", "noise", "third_cumulant")


def normalized_hermites(z, degree):
    """Normalized probabilists' Hermite polynomials, evaluated row-wise."""
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
    """Galerkin matrix of ``|v-g| (f(g,v)-f(v,g))``."""
    degree = int(degree)
    variance = float(variance)
    q = int(quadrature or (2 * degree + 4))
    tx, tw = hermgauss(q)
    yy, yw = laggauss(q)

    t = np.sqrt(2.0 * variance) * tx[None, :]
    blocks = []
    swaps = []
    weights = []
    for sign in (-1.0, 1.0):
        z = sign * np.sqrt(2.0 * variance * yy[:, None])
        v = (t + z) / np.sqrt(2.0)
        g = (t - z) / np.sqrt(2.0)
        hv = normalized_hermites(v.ravel() / np.sqrt(variance), degree)
        hg = normalized_hermites(g.ravel() / np.sqrt(variance), degree)
        blocks.append((hv[:, :, None] * hg[:, None, :]).reshape(q * q, -1))
        swaps.append((hg[:, :, None] * hv[:, None, :]).reshape(q * q, -1))
        weights.append((np.sqrt(variance) / np.pi * yw[:, None] * tw[None, :]).ravel())

    basis = np.concatenate(blocks)
    jump = np.concatenate(swaps) - basis
    weights = np.concatenate(weights)
    result = basis.T @ (weights[:, None] * jump)
    return (result + result.T) * 0.5


def _multiplication(degree, variance):
    """Return matrices for multiplication by v, g, v^2 and g^2."""
    width = degree + 1
    size = width * width
    V = np.zeros((size, size))
    G = np.zeros_like(V)
    for n in range(width):
        for m in range(width):
            j = n * width + m
            if n:
                V[(n - 1) * width + m, j] = np.sqrt(variance * n)
            if n + 1 < width:
                V[(n + 1) * width + m, j] = np.sqrt(variance * (n + 1))
            if m:
                G[n * width + m - 1, j] = np.sqrt(variance * m)
            if m + 1 < width:
                G[n * width + m + 1, j] = np.sqrt(variance * (m + 1))
    return V, G, V @ V, G @ G


def _derivative_g(degree, variance):
    width = degree + 1
    D = np.zeros((width * width, width * width))
    for n in range(width):
        for m in range(1, width):
            D[n * width + m - 1, n * width + m] = np.sqrt(m / variance)
    return D


def _state_operators(e, drag, degree):
    """Build the velocity/motor generator and heat tilt operators."""
    ta = float(e["temperature_a"])
    tb = float(e["temperature_b"])
    omega = float(e["angular_speed"])
    lam = float(e["relaxation_rate"])
    eta = float(e["flux_fraction"])
    flip = float(e["flip_rate"])

    variance = max(1.0, ta, tb, tb + omega * omega)
    width = degree + 1
    base_size = width * width
    size = 2 * base_size

    V0, G0, V20, G20 = _multiplication(degree, variance)
    D0 = _derivative_g(degree, variance)

    # Multiplication by the motor label s toggles the two s-parity blocks.
    S = np.zeros((size, size))
    for n in range(width):
        for m in range(width):
            q = n * width + m
            S[q + base_size, q] = 1.0
            S[q, q + base_size] = 1.0
    V = np.kron(np.eye(2), V0)
    G = np.kron(np.eye(2), G0)
    G2 = np.kron(np.eye(2), G20)
    D = np.kron(np.eye(2), D0)
    I = np.eye(size)

    L = np.zeros((size, size))
    for r in (0, 1):
        off = r * base_size
        for n in range(width):
            for m in range(width):
                j = off + n * width + m
                L[j, j] -= drag * n + lam * m
                if n >= 2:
                    L[off + (n - 2) * width + m, j] += (
                        drag * (ta / variance - 1.0) * np.sqrt(n * (n - 1))
                    )
                if m >= 2:
                    L[off + n * width + m - 2, j] += (
                        lam * (tb / variance - 1.0) * np.sqrt(m * (m - 1))
                    )

    Cabs = collision_weak_matrix(degree, variance)
    Cswap = np.zeros((base_size, base_size))
    for n in range(width):
        for m in range(width):
            j = n * width + m
            Cswap[m * width + n, j] += 1.0
            Cswap[j, j] -= 1.0
    C = (1.0 - eta) * Cswap + eta * Cabs
    L += RATE * np.kron(np.eye(2), C)

    # Flip generator: 0 on s-even functions and -2*flip on s-odd ones.
    # This is reflection of the label, not multiplication by the label
    # (the latter is S and is used by the flow-target terms below).
    F = np.kron(np.diag([1.0, -1.0]), np.eye(base_size))
    L += flip * (F - I)
    L += lam * omega * S @ D

    # For dQ=a dt+b dW, B1=a+sigma*b*d/dg and B2=b^2, with the
    # same W driving g.  This is the shared-Wiener correction.
    B1 = lam * (G2 - 2.0 * omega * S @ G + (omega * omega - tb) * I)
    B1 += -2.0 * lam * tb * G @ D + 2.0 * lam * tb * omega * S @ D
    B2 = 2.0 * lam * tb * (G2 - 2.0 * omega * S @ G + omega * omega * I)

    return L, V, B1, B2


def _stationary(L):
    """Stationary expectation functional p, with p[constant]=1."""
    n = L.shape[0]
    system = L.T.copy()
    rhs = np.zeros(n)
    system[0, :] = 0.0
    system[0, 0] = 1.0
    rhs[0] = 1.0
    return solve(system, rhs, check_finite=False)


def _solve_response(L, p, rhs):
    """Solve L x=rhs in the zero-mode subspace p*x=0."""
    system = L.copy()
    target = np.asarray(rhs, dtype=np.result_type(rhs, float)).copy()
    system[0, :] = p
    target[0] = 0.0
    return solve(system, target, check_finite=False)


def _cumulants(e, drag, degree=18, gated=False):
    """Return mean, variance-rate and third-cumulant-rate."""
    L, V, B1, B2 = _state_operators(e, drag, degree)
    p = _stationary(L)
    size = L.shape[0]
    one = np.zeros(size)
    one[0] = 1.0

    contrast = float(e.get("detector_contrast", 0.0)) if gated else 0.0
    harmonic = int(e.get("detector_harmonic", 1)) if gated else 1
    if not contrast:
        w = {0: 1.0}
        w2 = {0: 1.0}
    else:
        # Keys are Fourier indices in units of the detector harmonic.
        w = {0: 1.0, 1: contrast / 2.0, -1: contrast / 2.0}
        w2 = {
            0: 1.0 + contrast * contrast / 2.0,
            1: contrast,
            -1: contrast,
            2: contrast * contrast / 4.0,
            -2: contrast * contrast / 4.0,
        }

    def mode_operator(n):
        if n == 0:
            return L
        return L.astype(complex) + 1j * n * harmonic * V

    def response(n, rhs):
        if n == 0:
            return _solve_response(L, p, rhs)
        return solve(mode_operator(n), rhs, check_finite=False)

    def apply_fourier(coeffs, vectors, output_mode, operator):
        ans = np.zeros(size, dtype=complex)
        for mode, vec in vectors.items():
            ans += coeffs.get(output_mode - mode, 0.0) * (operator @ vec)
        return ans

    mean = float(np.real(p @ (w[0] * (B1 @ one))))

    # First response: (L_theta)r1 = mu1 - B1^w*1.
    r1 = {}
    for n in w:
        rhs = (mean * one if n == 0 else np.zeros(size)) - w[n] * (B1 @ one)
        r1[n] = response(n, rhs)

    term1 = np.zeros(size, dtype=complex)
    for m, vec in r1.items():
        term1 += w.get(-m, 0.0) * (B1 @ vec)
    mu2 = float(np.real(p @ (2.0 * term1 + w2[0] * (B2 @ one))))

    # Second response: (L_theta)r2 = mu2 + 2 mu1 r1 - 2 B1 r1 - B2*1.
    r2 = {}
    second_modes = set(w2)
    for m in r1:
        second_modes.update(m + q for q in w)
    for n in second_modes:
        rhs = (mu2 * one if n == 0 else np.zeros(size)).astype(complex)
        if n in r1:
            rhs += 2.0 * mean * r1[n]
        rhs -= 2.0 * apply_fourier(w, r1, n, B1)
        rhs -= w2.get(n, 0.0) * (B2 @ one)
        r2[n] = response(n, rhs)

    third_term = np.zeros(size, dtype=complex)
    for m, vec in r2.items():
        third_term += w.get(-m, 0.0) * (B1 @ vec)
    for m, vec in r1.items():
        third_term += w2.get(-m, 0.0) * (B2 @ vec)
    mu3 = float(np.real(p @ (3.0 * third_term)))
    return {"mean": mean, "noise": mu2, "third_cumulant": mu3}


def calibration_mean(e, drag):
    """Exact zero-flow, zero-flux calibration mean."""
    lam = float(e["relaxation_rate"])
    return drag * lam * RATE / (2.0 * drag * lam + RATE * (drag + lam)) * (
        float(e["temperature_a"]) - float(e["temperature_b"])
    )


def fit_drag(records):
    """Weighted one-parameter fit to the calibration records."""
    if not records:
        raise ValueError("at least one calibration record is required")

    def objective(drag):
        total = 0.0
        for record in records:
            e = record["input"]
            sigma = float(record.get("sigma", 1.0))
            total += ((calibration_mean(e, drag) - float(record["value"])) / sigma) ** 2
        return total

    result = minimize_scalar(objective, bounds=(0.4, 1.1), method="bounded",
                             options={"xatol": 1e-12})
    return float(np.clip(result.x, 0.4, 1.1))


@lru_cache(maxsize=512)
def _readout_cached(values, statistic, drag):
    e = dict(zip(FIELDS, values))
    if statistic not in STATISTICS:
        raise ValueError("unknown calorimeter statistic")
    if statistic == "mean" and values[0] == 0.0 and values[4] == 0.0:
        return float(calibration_mean(e, drag))
    return _cumulants(e, drag, degree=18)[statistic]


def readout(angular_speed, flip_rate, temperature_a, temperature_b,
            flux_fraction, relaxation_rate, statistic="mean", drag=0.67):
    values = tuple(float(x) for x in (
        angular_speed, flip_rate, temperature_a, temperature_b,
        flux_fraction, relaxation_rate,
    ))
    return _readout_cached(values, statistic, float(drag))


def predict_at(experiments, drag):
    result = []
    for experiment in experiments:
        statistic = experiment.get("statistic", "mean")
        contrast = float(experiment.get("detector_contrast", 0.0))
        if statistic not in STATISTICS:
            raise ValueError("unknown calorimeter statistic")
        if contrast:
            if float(experiment["angular_speed"]) != 0.0:
                raise ValueError("angularly gated preparations require zero flow target")
            result.append(_cumulants(experiment, drag, degree=18, gated=True)[statistic])
        else:
            result.append(readout(*(experiment[f] for f in FIELDS), statistic, drag))
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
