"""Stationary cumulants of the finite-contact rotor calorimeter.

The backward generator is represented in Gaussian Hermite polynomials of
t=(v+g)/sqrt(2), z=(v-g)/sqrt(2). A collision just reverses z. This makes
the relative-speed collision operator a set of small, exactly integrated
blocks, without replacing the retained contact by independent gas draws.

Cumulants come from Poisson equations for derivatives of the tilted
generator. For a gated detector, Fourier modes retain the actual angular
transport v*d/dtheta and the shared bath-B Wiener increment.
"""

from functools import lru_cache

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.special import roots_laguerre


RATE = 3.6
FIELDS = (
    "angular_speed", "flip_rate", "temperature_a", "temperature_b",
    "flux_fraction", "relaxation_rate",
)
STATISTICS = ("mean", "noise", "third_cumulant")


def normalized_hermites(z, degree):
    """Orthonormal probabilists' Hermite polynomials for a standard normal."""
    z = np.asarray(z).ravel()
    h = np.ones((len(z), degree + 1))
    if degree:
        h[:, 1] = z
    for n in range(1, degree):
        h[:, n + 1] = (z * h[:, n] - np.sqrt(n) * h[:, n - 1]) / np.sqrt(n + 1)
    return h


@lru_cache(maxsize=12)
def _basis(degree):
    """Dimensionless polynomial operators, with an exact degree cutoff.

    Products are formed two degrees beyond the cutoff before restriction.
    In particular, projecting multiplication by g twice must not discard the
    intermediate degree-(degree+1) term in multiplication by g squared.
    """
    pairs = [(n, m) for n in range(degree + 3) for m in range(degree + 3 - n)]
    lookup = {pair: j for j, pair in enumerate(pairs)}
    size = len(pairs)
    multiplications, derivatives = [], []
    for axis in (0, 1):
        rows, cols, vals = [], [], []
        dr, dc, dv = [], [], []
        for j, pair in enumerate(pairs):
            n = pair[axis]
            if n:
                lower = list(pair)
                lower[axis] -= 1
                i = lookup[tuple(lower)]
                rows.append(i); cols.append(j); vals.append(np.sqrt(n))
                dr.append(i); dc.append(j); dv.append(np.sqrt(n))
            upper = list(pair)
            upper[axis] += 1
            i = lookup.get(tuple(upper))
            if i is not None:
                rows.append(i); cols.append(j); vals.append(np.sqrt(n + 1))
        multiplications.append(sparse.csc_matrix((vals, (rows, cols)), shape=(size, size)))
        derivatives.append(sparse.csc_matrix((dv, (dr, dc)), shape=(size, size)))

    t, z = multiplications
    dt, dz = derivatives
    v, g = (t + z) / np.sqrt(2), (t - z) / np.sqrt(2)
    dv, dg = (dt + dz) / np.sqrt(2), (dt - dz) / np.sqrt(2)
    retained = np.array([j for j, pair in enumerate(pairs) if sum(pair) <= degree])

    def project(a):
        a = a.tocsc()[retained, :][:, retained].tocsc()
        a.eliminate_zeros()
        return a

    operators = tuple(project(a) for a in (v, g, dg, v @ dv, g @ dg, dv @ dv, dg @ dg, g @ g))
    pairs = np.array([pairs[j] for j in retained])
    size = len(pairs)
    odd = pairs[:, 1] % 2
    parity = (pairs[:, 0] + pairs[:, 1]) % 2

    # E[|v-g| H_m(z) H_j(z)] at unit reference variance.
    # With z=+/-sqrt(2*y), the odd-odd block is a polynomial Laguerre
    # integral. Even columns have zero collision jump.
    nodes, weights = roots_laguerre(degree + 2)
    h = normalized_hermites(np.sqrt(2 * nodes), degree)
    odd_degrees = np.arange(1, degree + 1, 2)
    h = h[:, odd_degrees]
    collision_block = -4 / np.sqrt(np.pi) * (h.T @ (weights[:, None] * h))
    rows, cols, vals = [], [], []
    for n in range(degree + 1):
        indices = np.flatnonzero((pairs[:, 0] == n) & (odd == 1))
        count = len(indices)
        rows.extend(np.repeat(indices, count))
        cols.extend(np.tile(indices, count))
        vals.extend(collision_block[:count, :count].ravel())
    flux = sparse.csc_matrix((vals, (rows, cols)), shape=(size, size))
    return operators, odd, parity, flux


def _operators(e, drag, degree):
    ta, tb = e["temperature_a"], e["temperature_b"]
    lam, eta = e["relaxation_rate"], e["flux_fraction"]
    omega, flip = e["angular_speed"], e["flip_rate"]
    variance = max(ta, tb)
    root = np.sqrt(variance)
    (v, g, dg, vdv, gdg, dv2, dg2, g2), odd, parity, flux = _basis(degree)
    identity = sparse.eye(len(odd), format="csc")

    # For ungated flow, use (s*v,s*g). A motor switch reverses both
    # coordinates, while the flow target becomes the constant omega.
    generator = (
        -drag * vdv - lam * gdg
        + drag * ta / variance * dv2 + lam * tb / variance * dg2
        + lam * omega / root * dg
        - sparse.diags(2 * flip * parity + 2 * RATE * (1 - eta) * odd)
    )
    if eta:
        generator = generator + RATE * eta * root * flux

    relative_square = variance * g2 - 2 * omega * root * g + omega**2 * identity
    # L(k) = L + k*A1 + k^2*A2/2 for E[exp(k*Q)]. The derivative
    # term in A1 is the covariance of dQ with the SAME noise driving g.
    first = lam * (relative_square - tb * identity) - 2 * lam * tb * (gdg - omega / root * dg)
    second = 2 * lam * tb * relative_square
    return generator.tocsc(), root * v, first.tocsc(), second.tocsc()


def _statistics(e, drag, degree, gated=False):
    parameters = dict(e)
    if e["angular_speed"] == 0:
        # The independent motor is immaterial when its two targets agree.
        # In the gated calculation these are laboratory velocities, so
        # including artificial sign flips would corrupt angular transport.
        parameters["flip_rate"] = 0.0
    generator, velocity, first, second = _operators(parameters, drag, degree)
    size = generator.shape[0]
    constant = np.zeros(size)
    constant[0] = 1.0

    # L*1=0. Removing the constant column/row permits one factorization
    # for stationarity and all zero-frequency Poisson equations.
    lu = splu(generator[1:, 1:])
    stationary = np.empty(size)
    stationary[0] = 1.0
    stationary[1:] = lu.solve(-generator[0, 1:].toarray().ravel(), trans="T")

    def poisson(rhs):
        solution = np.empty(size)
        solution[1:] = lu.solve(rhs[1:])
        solution[0] = -stationary[1:] @ solution[1:]
        return solution

    drift = first @ constant
    quadratic = second @ constant
    mean = float(stationary @ drift)
    r1 = poisson(mean * constant - drift)
    contrast = e.get("detector_contrast", 0.0) if gated else 0.0
    if not contrast:
        noise = float(stationary @ (quadratic + 2 * (first @ r1)))
        r2 = poisson(noise * constant + 2 * mean * r1 - 2 * (first @ r1) - quadratic)
        third = float(3 * stationary @ (first @ r2 + second @ r1))
        return np.array([mean, noise, third])

    # w has modes 0,+/-h; w^2 also has +/-2h. Only 0,+/-h response
    # modes can contribute to cumulants through order three. Negative
    # modes are complex conjugates. There is no phase-mixing closure.
    harmonic = e.get("detector_harmonic", 1)
    angular_lu = splu((generator + 1j * harmonic * velocity).tocsc())
    r1h = angular_lu.solve(-contrast / 2 * drift)
    w2mean = 1 + contrast**2 / 2
    noise = float(stationary @ (w2mean * quadratic + 2 * (first @ (r1 + contrast * r1h.real))))
    r2 = poisson(
        noise * constant + 2 * mean * r1
        - 2 * (first @ (r1 + contrast * r1h.real)) - w2mean * quadratic
    )
    r2h = angular_lu.solve(
        2 * mean * r1h - 2 * (first @ (r1h + contrast / 2 * r1)) - contrast * quadratic
    )
    third = float(3 * stationary @ (
        first @ (r2 + contrast * r2h.real)
        + second @ (w2mean * r1 + 2 * contrast * r1h.real)
    ))
    return np.array([mean, noise, third])


@lru_cache(maxsize=1024)
def _readouts(angular_speed, flip_rate, temperature_a, temperature_b,
              flux_fraction, relaxation_rate, contrast, harmonic, drag):
    e = dict(zip(FIELDS, (angular_speed, flip_rate, temperature_a, temperature_b,
                          flux_fraction, relaxation_rate)))
    e.update(detector_contrast=contrast, detector_harmonic=harmonic)
    if flux_fraction == 0 and contrast == 0:
        # Constant-clock collisions preserve polynomial degree. Degree six
        # therefore gives the first three cumulants exactly.
        return _statistics(e, drag, 6)

    previous = _statistics(e, drag, 24, gated=bool(contrast))
    for degree in (36, 48, 64, 96):
        current = _statistics(e, drag, degree, gated=bool(contrast))
        if np.allclose(current, previous, rtol=2e-6, atol=2e-7):
            return current
        previous = current
    return current


def calibration_mean(e, drag):
    lam = e["relaxation_rate"]
    return drag * lam * RATE / (2 * drag * lam + RATE * (drag + lam)) * (e["temperature_a"] - e["temperature_b"])


def fit_drag(records):
    """Weighted least squares for the exactly known calibration conductance."""
    records = list(records)
    if not records:
        raise ValueError("At least one calibration record is required.")
    lam = float(records[0]["input"]["relaxation_rate"])
    if any(r["input"]["relaxation_rate"] != lam for r in records):
        raise ValueError("Calibration records must share their relaxation rate.")
    delta = np.array([(r["input"]["temperature_a"] - r["input"]["temperature_b"]) / r["sigma"] for r in records])
    measured = np.array([r["value"] / r["sigma"] for r in records])
    if not delta @ delta > 0:
        raise ValueError("Calibration requires a nonzero temperature difference.")
    conductance = float(delta @ measured / (delta @ delta))
    # Conductance increases monotonically with drag; constrain before
    # inversion so even noisy out-of-range fits obey the specified bounds.
    bounds = [g * lam * RATE / (2 * g * lam + RATE * (g + lam)) for g in (0.4, 1.1)]
    conductance = np.clip(conductance, *bounds)
    drag = conductance * lam * RATE / (lam * RATE - conductance * (2 * lam + RATE))
    return float(np.clip(drag, 0.4, 1.1))


def predict_at(experiments, drag):
    result = []
    for e in experiments:
        statistic = e.get("statistic", "mean")
        if statistic not in STATISTICS:
            raise ValueError("Unknown calorimeter statistic: " + str(statistic))
        contrast = float(e.get("detector_contrast", 0.0))
        if contrast and e["angular_speed"] != 0:
            raise ValueError("Angularly gated preparations require zero flow target.")
        if statistic == "mean" and e["flux_fraction"] == 0 and e["angular_speed"] == 0:
            # Uniform stationary angle leaves the mean unchanged by gating.
            result.append(calibration_mean(e, drag))
            continue
        values = _readouts(
            *(float(e[f]) for f in FIELDS),
            contrast if statistic != "mean" else 0.0,
            int(e.get("detector_harmonic", 1)), float(drag),
        )
        result.append(values[STATISTICS.index(statistic)])
    return np.asarray(result, dtype=float)


class Model:
    def __init__(self):
        self.drag = None

    def fit(self, records):
        self.drag = fit_drag(records)
        return self

    def predict(self, experiments):
        if self.drag is None:
            raise ValueError("Fit the model before requesting predictions.")
        return predict_at(experiments, self.drag)
