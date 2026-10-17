"""SCGF for the weakly asymmetric exclusion process on a half-filled ring.

The current is counted over every bond and divided by the number of sites.  In
the weak-field limit, the homogeneous branch of the leading scaled SCGF is

    psi(E, lambda; D) = D / 4 * lambda * (lambda + 2 E).

At half filling that branch becomes unstable when
``lambda * (lambda + 2 * E) < -4*pi**2``.  The large-ring answer there is the
standard elliptic (phase-separated) branch.  In particular, the counting field
is a trajectory tilt, rather than an additional field in the jump rates.  A
common rate prefactor only rescales time, so it enters the answer linearly.
"""

import numpy as np


_D_MIN = 0.8
_D_MAX = 1.2
_INSTABILITY = -4.0 * np.pi ** 2


def _complete_elliptic_integrals(parameter):
    """Return complete elliptic K(m), E(m), using the AGM iteration.

    ``parameter`` is the conventional elliptic parameter m (not the modulus).
    The input range here is strictly below one, where this iteration is both
    fast and well conditioned.
    """
    a = 1.0
    b = float(np.sqrt(1.0 - parameter))
    c = float(np.sqrt(parameter))
    correction = 0.5 * c * c
    multiplier = 1.0

    for _ in range(50):
        c = 0.5 * (a - b)
        next_a = 0.5 * (a + b)
        b = float(np.sqrt(a * b))
        a = next_a
        correction += multiplier * c * c
        multiplier *= 2.0
        if abs(c) <= 2e-16 * a:
            break

    first = np.pi / (2.0 * a)
    second = first * (1.0 - correction)
    return first, second


def _phase_separated_branch(alpha):
    """Unit-diffusivity SCGF for alpha=sqrt(-lambda*(lambda+2E))."""
    # The profile modulus m is fixed by alpha = 4 K(m).  Bisection is ample
    # here: throughout the public domain alpha is at most 12.
    low, high = 0.0, 1.0 - 1e-15
    for _ in range(60):
        middle = 0.5 * (low + high)
        if 4.0 * _complete_elliptic_integrals(middle)[0] < alpha:
            low = middle
        else:
            high = middle
    parameter = 0.5 * (low + high)
    first, second = _complete_elliptic_integrals(parameter)
    return -0.25 * alpha * alpha * (2.0 * second / first - (1.0 - parameter))


def _unit_diffusivity_scgf(field, bias):
    """Evaluate psi at D=1, including the low-current phase branch."""
    invariant = bias * (bias + 2.0 * field)
    result = invariant / 4.0
    unstable = invariant < _INSTABILITY
    for index in np.flatnonzero(unstable):
        result[index] = _phase_separated_branch(float(np.sqrt(-invariant[index])))
    return result


def predict_at(experiments, diffusivity):
    """Evaluate the leading large-ring SCGF for an iterable of inputs."""
    # Building the arrays separately keeps the documented empty-list behaviour
    # (and the requested one-dimensional output shape) explicit.
    field = np.asarray([experiment["field"] for experiment in experiments], dtype=float)
    bias = np.asarray([experiment["bias"] for experiment in experiments], dtype=float)
    values = float(diffusivity) * _unit_diffusivity_scgf(field, bias)
    return np.asarray(values, dtype=float)


class Model:
    def __init__(self):
        self.diffusivity = 1.0

    def fit(self, records):
        """Fit ``D`` by its Gaussian-error maximum-likelihood estimate.

        With ``psi_i = D * x_i`` and known independent standard deviations,
        this is weighted least squares.  The physical parameter bounds are
        imposed after the unconstrained optimum is found.
        """
        records = list(records)
        if not records:
            # There is no calibration information, so retain a valid default.
            self.diffusivity = float(np.clip(self.diffusivity, _D_MIN, _D_MAX))
            return self

        field = np.asarray([record["input"]["field"] for record in records], dtype=float)
        bias = np.asarray([record["input"]["bias"] for record in records], dtype=float)
        observed = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)

        if not (np.isfinite(field).all() and np.isfinite(bias).all()
                and np.isfinite(observed).all() and np.isfinite(sigma).all()
                and np.all(sigma > 0.0)):
            raise ValueError("calibration records must contain finite values and positive sigma")

        coefficient = _unit_diffusivity_scgf(field, bias)
        weights = 1.0 / np.square(sigma)
        denominator = float(np.dot(weights, np.square(coefficient)))
        if denominator > 0.0 and np.isfinite(denominator):
            estimate = float(np.dot(weights * coefficient, observed) / denominator)
            if np.isfinite(estimate):
                self.diffusivity = float(np.clip(estimate, _D_MIN, _D_MAX))
                return self

        # Inputs at the symmetry points carry no information about D.  Keep a
        # finite physical value rather than producing a division-by-zero NaN.
        self.diffusivity = float(np.clip(self.diffusivity, _D_MIN, _D_MAX))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
