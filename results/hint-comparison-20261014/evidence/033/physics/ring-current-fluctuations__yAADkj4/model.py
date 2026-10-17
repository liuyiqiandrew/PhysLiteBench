"""SCGF and calibration interface for the ring-current problem.

The weak-asymmetry limit of the exclusion process has a useful variational
description. In units in which the ring coordinate has period one, let
``rho`` be a stationary density profile with mean one half and put
``a(rho) = rho*(1-rho)``. After optimizing over the (constant) current, the
dimensionless SCGF is

    max_rho { (lambda + E)^2 / int(1/a)
              - 1/4 int(rho'**2/a) - E**2 int(a) }.

The uniform profile gives lambda*(lambda+2*E)/4. It is not always the
maximizer: its first Fourier mode becomes unstable at
lambda*(lambda+2*E) = -4*pi**2. The non-uniform branch is evaluated by a
small Fourier variational problem below. This is also convenient because the
maximizing profile is only defined up to translation.
"""

from functools import lru_cache

import numpy as np


_TWO_PI = 2.0 * np.pi
_STABILITY_LIMIT = -4.0 * np.pi**2


def _uniform_value(field, bias):
    """Value for the spatially homogeneous profile, with D factored out."""
    return bias * (bias + 2.0 * field) / 4.0


@lru_cache(maxsize=512)
def _profile_value(field, bias):
    """Maximize the profile functional, with the diffusivity factored out.

    Write ``2*rho-1 = sin(phi)``. Profiles made from odd cosine modes have
    mean ``sin(phi)`` equal to zero, and include the reflection-symmetric
    maximizer. The nonlinear terms generate a rapidly convergent sequence of
    higher odd modes. Eight modes are already substantially more accurate than
    the precision of the calibration data.
    """
    field = float(field)
    bias = float(bias)
    homogeneous = _uniform_value(field, bias)
    k = bias * (bias + 2.0 * field)

    # The second variation of the uniform profile is proportional to
    # -(k + 4*pi**2). Avoid numerical optimization on the stable branch.
    if k >= _STABILITY_LIMIT - 1.0e-10:
        return homogeneous

    # The profile is phi(x) = sum c_n cos(2*pi*n*x), n odd. A 256 point
    # periodic quadrature resolves all retained modes and their nonlinear
    # products well; the result is checked against the uniform branch below.
    n_modes = 8
    n_grid = 256
    modes = np.arange(1, 2 * n_modes, 2, dtype=float)
    x = np.arange(n_grid, dtype=float) / n_grid
    basis = np.cos(_TWO_PI * modes[:, None] * x)
    derivative_basis = -_TWO_PI * modes[:, None] * np.sin(
        _TWO_PI * modes[:, None] * x
    )
    coefficient = (bias + field) ** 2 / 4.0
    field_sq = field * field

    def objective_and_gradient(c):
        phi = c @ basis
        phi_x = c @ derivative_basis
        cosine = np.cos(phi)

        # Profiles approaching rho=0 or 1 are not relevant in the public
        # range, but this guard keeps an optimizer trial point finite.
        if np.min(np.abs(cosine)) < 1.0e-7:
            return 1.0e6 + float(c @ c), np.zeros_like(c)

        secant_squared = 1.0 / (cosine * cosine)
        denominator = float(np.mean(secant_squared))
        value = (
            coefficient / denominator
            - 0.25 * float(np.mean(phi_x * phi_x))
            - field_sq / 4.0 * float(np.mean(cosine * cosine))
        )

        # Derivative of the value with respect to phi, followed by projection
        # onto the Fourier basis. scipy minimizes, hence the minus signs in
        # the returned pair.
        point_gradient = (
            -2.0
            * coefficient
            * secant_squared
            * np.tan(phi)
            / (denominator * denominator)
            + field_sq / 2.0 * cosine * np.sin(phi)
        )
        gradient = (
            basis @ point_gradient / n_grid
            - 0.5 * derivative_basis @ phi_x / n_grid
        )
        return -value, -gradient

    # scipy is available in the evaluation environment. Keep a conservative
    # first-harmonic fallback so prediction remains usable in minimal Python
    # installations too.
    try:
        from scipy.optimize import minimize
    except Exception:  # pragma: no cover - only for minimal installations
        return _first_harmonic_value(field, bias, coefficient, basis, derivative_basis)

    starts = [np.zeros(n_modes, dtype=float)]
    # Multiple amplitudes avoid relying on a particular basin of attraction
    # well inside the broken-symmetry phase.
    for amplitude in np.linspace(0.05, 1.5, 7):
        initial = np.zeros(n_modes, dtype=float)
        initial[0] = amplitude
        starts.append(initial)

    best = homogeneous
    for initial in starts:
        result = minimize(
            objective_and_gradient,
            initial,
            jac=True,
            method="BFGS",
            options={"maxiter": 500, "gtol": 1.0e-9},
        )
        candidate = -float(result.fun)
        if np.isfinite(candidate) and candidate > best:
            best = candidate

    return float(best)


def _first_harmonic_value(field, bias, coefficient, basis, derivative_basis):
    """Fallback profile maximization using only the first Fourier mode."""
    def value(amplitude):
        phi = amplitude * basis[0]
        phi_x = amplitude * derivative_basis[0]
        cosine = np.cos(phi)
        if np.min(np.abs(cosine)) < 1.0e-7:
            return -np.inf
        return (
            coefficient / np.mean(1.0 / (cosine * cosine))
            - 0.25 * np.mean(phi_x * phi_x)
            - field * field / 4.0 * np.mean(cosine * cosine)
        )

    # Golden-section search keeps this fallback dependency-free.
    left, right = 0.0, 1.55
    golden = (np.sqrt(5.0) - 1.0) / 2.0
    point_left = right - golden * (right - left)
    point_right = left + golden * (right - left)
    value_left, value_right = value(point_left), value(point_right)
    for _ in range(80):
        if value_left < value_right:
            left = point_left
            point_left, value_left = point_right, value_right
            point_right = left + golden * (right - left)
            value_right = value(point_right)
        else:
            right = point_right
            point_right, value_right = point_left, value_left
            point_left = right - golden * (right - left)
            value_left = value(point_left)
    return float(max(_uniform_value(field, bias), value(point_left), value(point_right)))


def predict_at(experiments, diffusivity):
    """Return raw asymptotic SCGF values for a sequence of inputs."""
    if not experiments:
        return np.empty(0, dtype=float)

    values = np.empty(len(experiments), dtype=float)
    for index, experiment in enumerate(experiments):
        field = float(experiment["field"])
        bias = float(experiment["bias"])
        values[index] = diffusivity * _profile_value(field, bias)
    return values


class Model:
    def __init__(self):
        self.diffusivity = 1.0

    def fit(self, records):
        """Fit D by weighted least squares, which is the Gaussian MLE."""
        if not records:
            self.diffusivity = 1.0
            return self

        inputs = [record["input"] for record in records]
        shape = np.array(
            [_profile_value(float(item["field"]), float(item["bias"])) for item in inputs],
            dtype=float,
        )
        observed = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)
        weights = 1.0 / (sigma * sigma)
        denominator = float(np.sum(weights * shape * shape))
        if not np.isfinite(denominator) or denominator <= 0.0:
            fitted = 1.0
        else:
            fitted = float(np.sum(weights * shape * observed) / denominator)
        self.diffusivity = float(np.clip(fitted, 0.8, 1.2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
