"""Stationary small-mass calorimetry for fast switched thermal contacts.

The valve and velocity both evolve on the mass time scale.  We solve their
coupled stationary moment equations at several small masses, perform the
required finite-mass background subtraction, and extrapolate to zero mass.
"""
from functools import lru_cache

import numpy as np
from scipy import optimize, sparse
from scipy.sparse.linalg import spsolve

INPUT_FIELDS = (
    "force_x", "force_y", "drag_ratio", "magnetic", "temperature",
    "contrast", "wavenumber", "bath_contrast", "switch_rate",
)


def _contacts(e, friction):
    """Temperature factors, drags, and column-stochastic valve generator."""
    rate = float(e["switch_rate"])
    if "valve_cycle" not in e:
        labels = np.array((1.0, -1.0))
        q = np.array(((-rate, rate), (rate, -rate)))
        scale = np.ones(2)
    else:
        labels = np.array((-1.0, 0.0, 1.0))
        bias = float(e["valve_cycle"])
        q = np.zeros((3, 3))
        for source in range(3):
            q[(source + 1) % 3, source] = rate * (1.0 + bias)
            q[(source - 1) % 3, source] = rate * (1.0 - bias)
            q[source, source] = -2.0 * rate
        scale = 1.0 + float(e.get("contact_drag_contrast", 0.0)) * labels
    gx = friction * scale
    return 1.0 + float(e["bath_contrast"]) * labels, gx, gx * float(e["drag_ratio"]), q


@lru_cache(maxsize=None)
def _derivative(n):
    """Sixth-order centred periodic first derivative on [0, 2*pi)."""
    h = 2.0 * np.pi / n
    offsets = (-3, -2, -1, 1, 2, 3)
    coeffs = (-1, 9, -45, 45, -9, 1)
    rows = np.repeat(np.arange(n), len(offsets))
    cols = np.asarray([[(row + offset) % n for offset in offsets]
                       for row in range(n)]).ravel()
    data = np.tile(np.asarray(coeffs, dtype=float) / (60.0 * h), n)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def _basis(max_degree):
    return [(p, total - p) for total in range(max_degree + 1) for p in range(total, -1, -1)]


def _background_coefficient(e, friction):
    """The 1/m coefficient of the matched uniform background entropy rate."""
    factors, gx, gy, q = _contacts(e, friction)
    contacts = len(gx)
    p0 = np.full(contacts, 1.0 / contacts)
    temp = float(e["temperature"]) * factors
    # Per-state components are xx, xy, yy.  This is the exact fast degree-2
    # system in the spatially uniform force-free background.
    matrix = np.zeros((3 * contacts, 3 * contacts))
    rhs = np.zeros(3 * contacts)
    magnetic = float(e["magnetic"])
    for s in range(contacts):
        for sp in range(contacts):
            for component in range(3):
                matrix[3 * s + component, 3 * sp + component] += q[s, sp]
        matrix[3 * s, 3 * s] -= 2.0 * gx[s]
        matrix[3 * s, 3 * s + 1] += 2.0 * magnetic
        rhs[3 * s] = -2.0 * gx[s] * temp[s] * p0[s]
        matrix[3 * s + 1, 3 * s + 1] -= gx[s] + gy[s]
        matrix[3 * s + 1, 3 * s + 2] += magnetic
        matrix[3 * s + 1, 3 * s] -= magnetic
        matrix[3 * s + 2, 3 * s + 2] -= 2.0 * gy[s]
        matrix[3 * s + 2, 3 * s + 1] -= 2.0 * magnetic
        rhs[3 * s + 2] = -2.0 * gy[s] * temp[s] * p0[s]
    moment = np.linalg.solve(matrix, rhs).reshape(contacts, 3)
    return float(np.sum(gx * (moment[:, 0] / temp - p0) + gy * (moment[:, 2] / temp - p0)))


def _stationary_moments(e, friction, epsilon, points=48, max_degree=6):
    """Stationary moments of u=sqrt(m)v, normalized as x densities."""
    factors, gx, gy, q = _contacts(e, friction)
    contacts = len(gx)
    pairs = _basis(max_degree)
    index = {pair: i for i, pair in enumerate(pairs)}
    degree_count = len(pairs)
    size = contacts * degree_count * points
    dx = 2.0 * np.pi / points
    phase = np.arange(points) * dx
    temp = float(e["temperature"]) * factors[:, None] * (
        1.0 + float(e["contrast"]) * np.cos(int(e["wavenumber"]) * phase)
    )
    derivative = _derivative(points)
    mat = sparse.lil_matrix((size, size))

    def sl(state, pair):
        start = (state * degree_count + index[pair]) * points
        return slice(start, start + points)

    identity = sparse.eye(points, format="lil")
    magnetic = float(e["magnetic"])
    fx, fy = float(e["force_x"]), float(e["force_y"])
    for s in range(contacts):
        for p, n in pairs:
            row = sl(s, (p, n))
            for source in range(contacts):
                if q[s, source]:
                    mat[row, sl(source, (p, n))] += q[s, source] * identity
            mat[row, sl(s, (p, n))] += (-p * gx[s] - n * gy[s]) * identity
            if p:
                mat[row, sl(s, (p - 1, n + 1))] += p * magnetic * identity
            if n:
                mat[row, sl(s, (p + 1, n - 1))] += -n * magnetic * identity
            if p >= 2:
                mat[row, sl(s, (p - 2, n))] += p * (p - 1) * gx[s] * sparse.diags(temp[s])
            if n >= 2:
                mat[row, sl(s, (p, n - 2))] += n * (n - 1) * gy[s] * sparse.diags(temp[s])
            if p + n < max_degree:
                mat[row, sl(s, (p + 1, n))] += -epsilon * derivative
            if p:
                mat[row, sl(s, (p - 1, n))] += epsilon * p * fx * identity
            if n:
                mat[row, sl(s, (p, n - 1))] += epsilon * n * fy * identity

    # One zeroth-moment equation is redundant because probability is conserved.
    anchor = (contacts - 1) * degree_count * points + points - 1
    mat[anchor, :] = 0.0
    for s in range(contacts):
        mat[anchor, sl(s, (0, 0))] = dx
    rhs = np.zeros(size)
    rhs[anchor] = 1.0
    solution = spsolve(mat.tocsc(), rhs)
    if not np.isfinite(solution).all():
        raise FloatingPointError("stationary moment solve failed")
    result = {pair: np.empty((contacts, points)) for pair in pairs}
    for s in range(contacts):
        for pair in pairs:
            result[pair][s] = solution[sl(s, pair)]
    return phase, dx, temp, result


def _one_mass_excess(e, friction, epsilon):
    phase, dx, temp, moment = _stationary_moments(e, friction, epsilon)
    _, gx, gy, _ = _contacts(e, friction)
    if "detector_phase" in e:
        weight = (1.0 + np.cos(phase - float(e["detector_phase"]))) / 2.0
    else:
        weight = np.ones_like(phase)
    local = gx[:, None] * (moment[(2, 0)] / temp - moment[(0, 0)])
    local += gy[:, None] * (moment[(0, 2)] / temp - moment[(0, 0)])
    rate_coefficient = dx * np.sum(weight * np.sum(local, axis=0))
    residence = dx * np.sum(weight * np.sum(moment[(0, 0)], axis=0))
    return (rate_coefficient - _background_coefficient(e, friction) * residence) / epsilon ** 2


@lru_cache(maxsize=4096)
def _entropy_cached(args):
    e = dict(zip(INPUT_FIELDS, args[:len(INPUT_FIELDS)]))
    friction = args[len(INPUT_FIELDS)]
    detector_phase, valve_cycle, drag_contrast = args[len(INPUT_FIELDS) + 1:]
    if detector_phase is not None:
        e["detector_phase"] = detector_phase
    if valve_cycle is not None:
        e["valve_cycle"] = valve_cycle
        e["contact_drag_contrast"] = drag_contrast
    # This stationary observable is even in epsilon=sqrt(m).  Extrapolating
    # after background subtraction avoids ever interpreting its divergent terms.
    epsilons = np.array((0.018, 0.030, 0.050))
    finite_mass = np.array([_one_mass_excess(e, friction, eps) for eps in epsilons])
    return float(np.polynomial.polynomial.polyfit(epsilons ** 2, finite_mass, 2)[0])


def entropy_rate(*args):
    return _entropy_cached(tuple(args))


def coefficients(experiments, friction=1.0):
    return np.asarray([
        entropy_rate(*(e[key] for key in INPUT_FIELDS), float(friction),
                     e.get("detector_phase"), e.get("valve_cycle"),
                     e.get("contact_drag_contrast", 0.0))
        for e in experiments
    ], dtype=float)


class Model:
    def __init__(self):
        self.friction = None

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")
        experiments = [record["input"] for record in records]
        observed = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)
        if not np.isfinite(observed).all() or not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError("calibration values and sigmas must be finite and positive")

        def objective(friction):
            residual = (coefficients(experiments, float(friction)) - observed) / sigma
            return float(residual @ residual)

        result = optimize.minimize_scalar(objective, bounds=(0.7, 1.6), method="bounded",
                                          options={"xatol": 2e-5})
        self.friction = float(result.x)
        return self

    def predict(self, experiments):
        if self.friction is None:
            raise RuntimeError("fit must be called before predict")
        return coefficients(experiments, self.friction)
