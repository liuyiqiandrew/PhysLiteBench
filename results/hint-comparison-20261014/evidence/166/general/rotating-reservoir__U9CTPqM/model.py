"""Stationary calorimeter readouts for the finite retained-contact model.

The velocity process is solved with a Hermite--Galerkin representation.  This
keeps the two motor states explicitly and derives heat cumulants from the
tilted generator, including the Wiener noise shared by gas and calorimeter.
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
    """Orthonormal probabilists' Hermite polynomials at *x*."""
    x = np.asarray(x)
    values = np.empty((x.size, degree + 1)); values[:, 0] = 1.0
    if degree: values[:, 1] = x
    for n in range(1, degree):
        values[:, n + 1] = (x * values[:, n] - np.sqrt(n) * values[:, n - 1]) / np.sqrt(n + 1)
    return values


@lru_cache(maxsize=16)
def _collision_matrix(degree, variance, quadrature):
    """Weak matrix of ``|v-g| (swap-I)`` in a Gaussian Hermite basis."""
    x, xw = hermgauss(quadrature); y, yw = laggauss(quadrature)
    t = np.sqrt(2.0 * variance) * x[None, :]
    blocks, swapped, weights = [], [], []
    for sign in (-1.0, 1.0):
        z = sign * np.sqrt(2.0 * variance * y[:, None])
        v, g = (t + z) / np.sqrt(2.0), (t - z) / np.sqrt(2.0)
        hv = _hermites((v / np.sqrt(variance)).ravel(), degree)
        hg = _hermites((g / np.sqrt(variance)).ravel(), degree)
        blocks.append((hv[:, :, None] * hg[:, None, :]).reshape(quadrature * quadrature, -1))
        swapped.append((hg[:, :, None] * hv[:, None, :]).reshape(quadrature * quadrature, -1))
        # Gaussian density for t, and |v-g| times Gaussian density for z.
        weights.append((np.sqrt(variance) / np.pi * yw[:, None] * xw[None, :]).ravel())
    basis = np.concatenate(blocks); jump = np.concatenate(swapped) - basis
    result = basis.T @ (np.concatenate(weights)[:, None] * jump)
    return (result + result.T) * 0.5  # reversible swap generator


def _indices(degree):
    return [(n, m) for n in range(degree + 1) for m in range(degree + 1 - n)]


def _operators(e, drag, degree):
    """Backward generator and its first two heat-tilt derivatives.

    For ``E exp(k Q)``: L(k) = L + k*A1 + k**2*A2/2.
    Matrix columns are input basis functions and rows output functions.
    """
    ta, tb = float(e["temperature_a"]), float(e["temperature_b"])
    omega, lam = float(e["angular_speed"]), float(e["relaxation_rate"])
    eta, flip = float(e["flux_fraction"]), float(e["flip_rate"])
    variance = max(ta, tb, 0.35)
    pairs = _indices(degree); local = {pair: i for i, pair in enumerate(pairs)}
    width, nlocal, size = degree + 1, len(pairs), 2 * len(pairs)
    loc = lambda state, n, m: state * nlocal + local[(n, m)]
    matrix = np.zeros((size, size)); multiply_g = np.zeros_like(matrix)
    derivative_g = np.zeros_like(matrix); square_g = np.zeros_like(matrix)

    def put(array, row, col, value):
        array[row, col] += value

    for state, sign in enumerate((-1.0, 1.0)):
        flow = sign * omega
        for n, m in pairs:
            col = loc(state, n, m)
            matrix[col, col] -= drag * n + lam * m + flip
            if n >= 2:
                put(matrix, loc(state, n - 2, m), col,
                    drag * (ta / variance - 1.0) * np.sqrt(n * (n - 1)))
            if m >= 2:
                put(matrix, loc(state, n, m - 2), col,
                    lam * (tb / variance - 1.0) * np.sqrt(m * (m - 1)))
            if m >= 1:
                put(matrix, loc(state, n, m - 1), col, lam * flow * np.sqrt(m / variance))
            put(matrix, loc(1 - state, n, m), col, flip)
            if m >= 1:
                put(multiply_g, loc(state, n, m - 1), col, np.sqrt(variance * m))
                put(derivative_g, loc(state, n, m - 1), col, np.sqrt(m / variance))
            if (n, m + 1) in local:
                put(multiply_g, loc(state, n, m + 1), col, np.sqrt(variance * (m + 1)))
            put(square_g, col, col, variance * (2 * m + 1))
            if m >= 2: put(square_g, loc(state, n, m - 2), col, variance * np.sqrt(m * (m - 1)))
            if (n, m + 2) in local: put(square_g, loc(state, n, m + 2), col, variance * np.sqrt((m + 1) * (m + 2)))

    full_collision = _collision_matrix(degree, variance, max(2 * degree + 6, 36))
    selected = [n * width + m for n, m in pairs]
    collision = full_collision[np.ix_(selected, selected)]
    for state in range(2):
        sl = slice(state * nlocal, (state + 1) * nlocal)
        matrix[sl, sl] += RATE * eta * collision
        matrix[sl, sl] -= RATE * (1.0 - eta) * np.eye(nlocal)
        for n, m in pairs:
            matrix[loc(state, m, n), loc(state, n, m)] += RATE * (1.0 - eta)

    a1 = np.zeros_like(matrix); a2 = np.zeros_like(matrix)
    for state, sign in enumerate((-1.0, 1.0)):
        flow = sign * omega; sl = slice(state * nlocal, (state + 1) * nlocal)
        g, dg, g2 = multiply_g[sl, sl], derivative_g[sl, sl], square_g[sl, sl]
        relative_square = g2 - 2.0 * flow * g + flow * flow * np.eye(nlocal)
        # a + b*sigma*d/dg and b^2; b shares the gas thermostat Wiener noise.
        a1[sl, sl] = lam * (relative_square - tb * np.eye(nlocal)) - 2.0 * lam * tb * (g - flow * np.eye(nlocal)) @ dg
        a2[sl, sl] = 2.0 * lam * tb * relative_square

    constant = np.zeros(size); constant[loc(0, 0, 0)] = constant[loc(1, 0, 0)] = 1.0
    stationary_system = matrix.T.copy(); stationary_system[0] = constant
    target = np.zeros(size); target[0] = 1.0
    probability = solve(stationary_system, target, check_finite=False)
    return matrix, a1, a2, probability, constant


def _rates(e, drag, a2_factor=1.0):
    """First three stationary cumulant rates of the requested heat signal."""
    # The absolute-speed channel is non-polynomial.  Degree 32 gives stable
    # readouts even at the edges of the documented temperature/rate ranges.
    degree = 6 if float(e["flux_fraction"]) == 0.0 else 32
    matrix, a1, a2, p, one = _operators(e, drag, degree)
    a2 *= a2_factor
    mean = float(p @ (a1 @ one))
    system = matrix.copy(); system[0] = p
    rhs1 = mean * one - a1 @ one; rhs1[0] = 0.0
    r1 = solve(system, rhs1, check_finite=False)
    noise = float(p @ (a2 @ one) + 2.0 * p @ (a1 @ r1))
    rhs2 = noise * one + 2.0 * mean * r1 - 2.0 * (a1 @ r1) - a2 @ one; rhs2[0] = 0.0
    r2 = solve(system, rhs2, check_finite=False)
    third = float(p @ (3.0 * a1 @ r2 + 3.0 * a2 @ r1))
    return {"mean": mean, "noise": noise, "third_cumulant": third}


def calibration_mean(e, drag):
    """Exact mean heat rate for the zero-flow, constant-clock calibration."""
    lam = float(e["relaxation_rate"])
    return (drag * lam * RATE / (2.0 * drag * lam + RATE * (drag + lam))
            * (float(e["temperature_a"]) - float(e["temperature_b"])))


def fit_drag(records):
    if not records: raise ValueError("at least one calibration record is required")
    def objective(drag):
        total = 0.0
        for record in records:
            sigma = float(record["sigma"])
            if not np.isfinite(sigma) or sigma <= 0: raise ValueError("calibration sigma must be positive and finite")
            total += ((calibration_mean(record["input"], drag) - float(record["value"])) / sigma) ** 2
        return total
    return float(minimize_scalar(objective, bounds=(0.4, 1.1), method="bounded", options={"xatol": 1e-12}).x)


@lru_cache(maxsize=512)
def _readout(angular_speed, flip_rate, temperature_a, temperature_b, flux_fraction,
             relaxation_rate, statistic, contrast, harmonic, drag):
    if statistic not in {"mean", "noise", "third_cumulant"}: raise ValueError("unknown calorimeter statistic")
    if contrast and angular_speed != 0.0: raise ValueError("Angularly gated preparations require zero flow target.")
    e = dict(zip(FIELDS, (angular_speed, flip_rate, temperature_a, temperature_b, flux_fraction, relaxation_rate)))
    if contrast:
        # Documented independently rapid detector-phase approximation:
        # <w>=1 and <w^2>=1+contrast^2/2.  Harmonic drops out in this limit.
        return _rates(e, drag, 1.0 + contrast * contrast / 2.0)[statistic]
    if statistic == "mean" and flux_fraction == 0.0 and angular_speed == 0.0:
        return calibration_mean(e, drag)
    return _rates(e, drag)[statistic]


def predict_at(experiments, drag):
    if drag is None: raise RuntimeError("fit must be called before predict")
    output = []
    for experiment in experiments:
        try: values = [float(experiment[field]) for field in FIELDS]
        except KeyError as exc: raise ValueError("missing experiment field: %s" % exc.args[0]) from exc
        output.append(_readout(*values, experiment.get("statistic", "mean"),
                               float(experiment.get("detector_contrast", 0.0)),
                               int(experiment.get("detector_harmonic", 1)), float(drag)))
    return np.asarray(output, dtype=float)


class Model:
    def __init__(self): self.drag = None

    def fit(self, records):
        self.drag = fit_drag(records)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.drag)
