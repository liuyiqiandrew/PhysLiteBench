"""Stationary calorimeter predictions for the finite-contact rotor model.

The calculation below is a Galerkin discretisation of the *backward*
generator.  Normalised Hermite polynomials are particularly convenient here:
the Ornstein--Uhlenbeck pieces are sparse, while the only non-polynomial term
in the generator (the relative-speed collision clock) can be projected once
with Gaussian quadrature.  Cumulants are obtained by differentiating the
tilted generator, rather than by simulating long stochastic trajectories.
"""

from functools import lru_cache

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.laguerre import laggauss
from scipy.linalg import lu_factor, lu_solve


RATE = 3.6
FIELDS = (
    "angular_speed",
    "flip_rate",
    "temperature_a",
    "temperature_b",
    "flux_fraction",
    "relaxation_rate",
)


def normalized_hermites(x, degree):
    """Values of orthonormal probabilists' Hermites through ``degree``."""
    x = np.asarray(x)
    values = np.empty((x.size, degree + 1), dtype=float)
    values[:, 0] = 1.0
    if degree:
        values[:, 1] = x
    for n in range(1, degree):
        values[:, n + 1] = (
            x * values[:, n] - np.sqrt(n) * values[:, n - 1]
        ) / np.sqrt(n + 1)
    return values


@lru_cache(maxsize=4)
def _unit_collision_weak_matrix(degree):
    """Projection of ``|v-g| [f(g,v)-f(v,g)]`` onto a Hermite basis.

    In centre/relative coordinates the absolute value cancels the square-root
    Jacobian after putting the relative coordinate through a Laguerre rule.
    This is both more accurate at ``v == g`` and substantially less noisy than
    sampling the collision clock.
    """
    # Products of two retained rectangular-basis functions need at most this
    # polynomial order in the Hermite and Laguerre rules.
    variance = 1.0
    nq = 2 * degree + 5
    hx, hw = hermgauss(nq)
    ly, lw = laggauss(nq)

    centre = np.sqrt(2.0 * variance) * hx[None, :]
    blocks = []
    swapped = []
    weights = []
    for sign in (-1.0, 1.0):
        relative = sign * np.sqrt(2.0 * variance * ly[:, None])
        v = (centre + relative) / np.sqrt(2.0)
        g = (centre - relative) / np.sqrt(2.0)
        hv = normalized_hermites(v.ravel() / np.sqrt(variance), degree)
        hg = normalized_hermites(g.ravel() / np.sqrt(variance), degree)
        blocks.append((hv[:, :, None] * hg[:, None, :]).reshape(nq * nq, -1))
        swapped.append((hg[:, :, None] * hv[:, None, :]).reshape(nq * nq, -1))
        # This is the joint N(0, variance) expectation including |v-g|.
        weights.append((np.sqrt(variance) / np.pi * lw[:, None] * hw[None, :]).ravel())

    basis = np.concatenate(blocks, axis=0)
    jump = np.concatenate(swapped, axis=0) - basis
    weight = np.concatenate(weights, axis=0)
    matrix = basis.T @ (weight[:, None] * jump)
    # The swap clock is reversible for the reference Gaussian.  Symmetrising
    # only removes quadrature round-off and keeps later linear solves stable.
    return 0.5 * (matrix + matrix.T)


def collision_weak_matrix(degree, variance):
    """Collision projection at a Gaussian reference variance.

    Rescaling v and g by ``sqrt(variance)`` leaves Hermite coordinates and
    the velocity exchange unchanged; the clock itself contributes the sole
    factor of ``sqrt(variance)``.  One unit-variance quadrature matrix can
    consequently serve all experimental temperatures and flow settings.
    """
    return np.sqrt(float(variance)) * _unit_collision_weak_matrix(degree)


def _basis(degree):
    pairs = [(n, m) for n in range(degree + 1) for m in range(degree + 1 - n)]
    lookup = {pair: i for i, pair in enumerate(pairs)}
    return pairs, lookup


def _stationary_left_vector(generator, nmodes):
    """Return p such that p L = 0 and p applied to constant one is one."""
    size = generator.shape[0]
    system = generator.T.copy()
    rhs = np.zeros(size)
    # The motor is an independent symmetric two-state chain, so either
    # component has stationary probability one half.  This fixes the nullspace
    # without adding a nearly dependent all-constant row.
    system[0, :] = 0.0
    system[0, 0] = 1.0
    rhs[0] = 0.5
    p = lu_solve(lu_factor(system, check_finite=False), rhs, check_finite=False)
    constant = np.zeros(size)
    constant[0] = 1.0
    constant[nmodes] = 1.0
    return p / float(p @ constant)


def _replace_null_row(generator, p):
    """A nonsingular version of L imposing p r = 0 on Poisson solutions."""
    system = generator.copy()
    system[0, :] = p
    return lu_factor(system, check_finite=False)


def _operators(e, drag, degree):
    """Build L, heat-tilt derivatives and the stationary functional."""
    ta = float(e["temperature_a"])
    tb = float(e["temperature_b"])
    u = float(e["angular_speed"])
    flip = float(e["flip_rate"])
    eta = float(e["flux_fraction"])
    lam = float(e["relaxation_rate"])

    # A width close to the conditional velocity second moment improves the
    # truncated representation of the two motor-conditioned distributions.
    variance = max(ta, tb + u * u, 0.5)
    pairs, lookup = _basis(degree)
    nmodes = len(pairs)
    size = 2 * nmodes

    def index(state, n, m):
        return state * nmodes + lookup[(n, m)]

    generator = np.zeros((size, size), dtype=float)

    # The direct g, d/dg, and g^2 operators are needed for the tilted
    # generator.  g^2 is constructed directly: multiplying two truncated g
    # matrices loses a boundary contribution to its diagonal.
    gmul = np.zeros_like(generator)
    dg = np.zeros_like(generator)
    g2 = np.zeros_like(generator)

    for state, sign in enumerate((1.0, -1.0)):
        other = 1 - state
        for jmode, (n, m) in enumerate(pairs):
            j = state * nmodes + jmode
            generator[j, j] -= drag * n + lam * m + flip + RATE * (1.0 - eta)
            generator[other * nmodes + jmode, j] += flip

            # Constant-rate exchange.
            generator[index(state, m, n), j] += RATE * (1.0 - eta)

            if n >= 2:
                generator[index(state, n - 2, m), j] += (
                    drag * (ta / variance - 1.0) * np.sqrt(n * (n - 1))
                )
            if m >= 2:
                generator[index(state, n, m - 2), j] += (
                    lam * (tb / variance - 1.0) * np.sqrt(m * (m - 1))
                )
            if m >= 1:
                generator[index(state, n, m - 1), j] += (
                    lam * sign * u * np.sqrt(m / variance)
                )

            if m >= 1:
                gmul[index(state, n, m - 1), j] += np.sqrt(variance * m)
                dg[index(state, n, m - 1), j] += np.sqrt(m / variance)
            if (n, m + 1) in lookup:
                gmul[index(state, n, m + 1), j] += np.sqrt(variance * (m + 1))

            g2[j, j] += variance * (2 * m + 1)
            if m >= 2:
                g2[index(state, n, m - 2), j] += variance * np.sqrt(m * (m - 1))
            if (n, m + 2) in lookup:
                g2[index(state, n, m + 2), j] += variance * np.sqrt((m + 1) * (m + 2))

    # The flux-clock contribution is diagonal in the motor state.  Avoiding
    # its quadrature construction for eta=0 also keeps purely OU readouts
    # inexpensive.
    if eta:
        collision = collision_weak_matrix(degree, float(variance))
        rectangular = degree + 1
        retained = np.array([n * rectangular + m for n, m in pairs])
        collision = collision[np.ix_(retained, retained)]
        for state in range(2):
            sl = slice(state * nmodes, (state + 1) * nmodes)
            generator[sl, sl] += RATE * eta * collision

    identity = np.eye(size)
    constant = np.zeros(size)
    constant[0] = 1.0
    constant[nmodes] = 1.0

    # For dQ = a dt + b dW, tilting E exp(k Q) adds
    # k[a + b sigma_g d/dg] + k^2 b^2/2.  These are its first and second
    # derivatives at k=0, respectively.
    a1 = np.zeros_like(generator)
    a2 = np.zeros_like(generator)
    gd = gmul @ dg
    for state, sign in enumerate((1.0, -1.0)):
        sl = slice(state * nmodes, (state + 1) * nmodes)
        x2 = g2[sl, :] - 2.0 * sign * u * gmul[sl, :] + u * u * identity[sl, :]
        xdg = gd[sl, :] - sign * u * dg[sl, :]
        a1[sl, :] = lam * (x2 - tb * identity[sl, :]) - 2.0 * lam * tb * xdg
        a2[sl, :] = 2.0 * lam * tb * x2

    p = _stationary_left_vector(generator, nmodes)
    return generator, a1, a2, p, constant


def _cumulants(e, drag, statistic, detector_contrast=0.0):
    """Mean, noise, or third cumulant rate for one physical preparation."""
    if statistic not in {"mean", "noise", "third_cumulant"}:
        raise ValueError("unknown calorimeter statistic")

    # The cusp in the flux clock gives algebraic, rather than geometric,
    # spectral convergence.  Degree 30 is a useful accuracy/cost balance for
    # it.  With a purely constant collision clock, all equations needed up to
    # the third cumulant close at degree six.
    degree = 6 if float(e["flux_fraction"]) == 0.0 else 30
    generator, a1, a2, p, constant = _operators(e, drag, degree=degree)
    mean = float(p @ (a1 @ constant))
    if statistic == "mean":
        return mean

    # In the documented rapid independent-phase limit, w has E[w]=1 and
    # E[w^2]=1+contrast^2/2.  Thus only the quadratic tilt term changes.
    a2 = a2 * (1.0 + 0.5 * detector_contrast * detector_contrast)
    factor = _replace_null_row(generator, p)

    rhs1 = mean * constant - a1 @ constant
    rhs1[0] = 0.0
    r1 = lu_solve(factor, rhs1, check_finite=False)
    noise = float(p @ (a2 @ constant + 2.0 * a1 @ r1))
    if statistic == "noise":
        return noise

    rhs2 = noise * constant + 2.0 * mean * r1 - 2.0 * a1 @ r1 - a2 @ constant
    rhs2[0] = 0.0
    r2 = lu_solve(factor, rhs2, check_finite=False)
    return float(p @ (3.0 * a1 @ r2 + 3.0 * a2 @ r1))


def calibration_mean(e, drag):
    """Exact zero-flow, constant-clock calibration conductance."""
    lam = float(e["relaxation_rate"])
    return (
        drag
        * lam
        * RATE
        / (2.0 * drag * lam + RATE * (drag + lam))
        * (float(e["temperature_a"]) - float(e["temperature_b"]))
    )


def fit_drag(records):
    """Weighted least-squares estimate of the shared laboratory drag."""
    if not records:
        raise ValueError("at least one calibration record is required")
    # Calibration has Q = conductance * (Ta-Tb), so fitting the conductance is
    # a one-parameter weighted linear regression.  The inverse relation is
    # monotone on the allowed drag interval.
    x = np.asarray(
        [
            (float(r["input"]["temperature_a"]) - float(r["input"]["temperature_b"]))
            / float(r["sigma"])
            for r in records
        ]
    )
    y = np.asarray([float(r["value"]) / float(r["sigma"]) for r in records])
    denom = float(x @ x)
    if denom == 0.0:
        raise ValueError("calibration records do not identify drag")
    conductance = float(x @ y / denom)
    # All supplied calibration records use a common relaxation rate.  Use the
    # first one; retaining this explicit check prevents silently fitting an
    # inapplicable closed form to arbitrary records.
    lam = float(records[0]["input"]["relaxation_rate"])
    if not all(float(r["input"]["relaxation_rate"]) == lam for r in records):
        raise ValueError("calibration relaxation_rate must be common")
    drag = conductance * lam * RATE / (lam * RATE - conductance * (2.0 * lam + RATE))
    return float(np.clip(drag, 0.4, 1.1))


@lru_cache(maxsize=4096)
def _cached_readout(values, statistic, drag, contrast):
    e = dict(zip(FIELDS, values))
    # Retain the analytic expression for calibration-style inputs.  It is exact
    # and avoids numerical projection error in Model.fit's own records.
    if (
        statistic == "mean"
        and contrast == 0.0
        and e["flux_fraction"] == 0.0
        and e["angular_speed"] == 0.0
    ):
        return calibration_mean(e, drag)
    return _cumulants(e, drag, statistic, contrast)


def predict_at(experiments, drag):
    if drag is None:
        raise RuntimeError("fit must be called before predict")
    result = []
    for experiment in experiments:
        contrast = float(experiment.get("detector_contrast", 0.0))
        if contrast and float(experiment["angular_speed"]) != 0.0:
            raise ValueError("Angularly gated preparations require zero flow target.")
        statistic = experiment.get("statistic", "mean")
        # detector_harmonic deliberately does not enter the rapid-mixing limit:
        # every allowed nonzero harmonic has the same phase averages.
        values = tuple(float(experiment[name]) for name in FIELDS)
        result.append(_cached_readout(values, statistic, float(drag), contrast))
    return np.asarray(result, dtype=float)


class Model:
    def __init__(self):
        self.drag = None

    def fit(self, records):
        self.drag = fit_drag(records)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.drag)
