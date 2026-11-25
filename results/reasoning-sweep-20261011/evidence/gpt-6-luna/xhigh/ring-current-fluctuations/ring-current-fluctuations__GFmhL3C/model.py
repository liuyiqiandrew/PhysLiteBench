"""Current SCGF model and calibration interface."""
from functools import lru_cache

import numpy as np


def _traveling_profile_value(field, bias):
    """Optimize the traveling-profile MFT branch, in units of D.

    For a profile ``rho(x)`` moving at speed ``v``, continuity gives the
    macroscopic current ``q + v*rho``.  Optimizing over its mean current and
    speed leaves a one-dimensional periodic profile variational problem.  We
    discretize that smooth problem in the angle ``rho=(1+cos(theta))/2``;
    this angle makes the gradient cost simply ``(theta')**2/4``.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        # NumPy-only environments can still use the uniform solution.
        return bias * (bias + 2.0 * field) / 4.0

    # A finer mesh resolves the narrow interfaces at the largest public fields.
    n = 96
    x = np.arange(n, dtype=float) / n
    k = bias + field
    eps = 1e-3
    bounds = [(eps, np.pi - eps)] * n

    def value_and_gradient(theta):
        sine = np.sin(theta)
        cosine = np.cos(theta)
        chi = 0.25 * sine * sine
        phi = 0.5 * cosine
        chi_prime = 0.5 * sine * cosine
        phi_prime = -0.5 * sine

        if abs(k) > 1e-14:
            inv_chi = 1.0 / chi
            inv_chi_prime = -chi_prime / (chi * chi)
            phi_over_chi_prime = (
                phi_prime / chi - phi * chi_prime / (chi * chi)
            )
            phi2_over_chi_prime = (
                2.0 * phi * phi_prime / chi
                - phi * phi * chi_prime / (chi * chi)
            )
            i0 = float(np.mean(inv_chi))
            i1 = float(np.mean(phi * inv_chi))
            i2 = float(np.mean(phi * phi * inv_chi))
            if i2 > 1e-10:
                h = i0 - i1 * i1 / i2
                h_prime = (
                    inv_chi_prime
                    - 2.0 * i1 / i2 * phi_over_chi_prime
                    + i1 * i1 / (i2 * i2) * phi2_over_chi_prime
                ) / n
            else:
                h = i0
                h_prime = inv_chi_prime / n
            if not np.isfinite(h) or h <= 0:
                h = i0
                h_prime = inv_chi_prime / n
            current_term = k * k / h
            current_gradient = -k * k * h_prime / (h * h)
        else:
            current_term = 0.0
            current_gradient = np.zeros_like(theta)

        differences = np.diff(np.r_[theta, theta[0]])
        gradient_cost = 0.25 * n * float(np.dot(differences, differences))
        value = current_term - field * field * float(np.mean(chi)) - gradient_cost
        gradient = (
            current_gradient
            - field * field * chi_prime / n
            - n * (theta - 0.5 * np.roll(theta, 1) - 0.5 * np.roll(theta, -1))
        )
        return value, gradient

    def value(theta):
        return value_and_gradient(theta)[0]

    def objective_gradient(theta):
        return -value_and_gradient(theta)[1]

    def density_constraint(theta):
        return float(np.mean(np.cos(theta)))

    constraint = {"type": "eq", "fun": density_constraint}
    candidates = []
    # The flat profile remains a candidate even after it loses local stability.
    flat_value = bias * (bias + 2.0 * field) / 4.0
    candidates.append(flat_value)

    # Multi-start over the fundamental wave and a few asymmetric deformations.
    # The latter permit the optimum wave to select its own propagation speed.
    starts = []
    for amplitude in (0.35, 0.8, 1.35):
        starts.append(np.pi / 2.0 + amplitude * np.cos(2.0 * np.pi * x))
    for amplitude in (0.65, 1.2):
        starts.append(
            np.pi / 2.0
            + amplitude * np.cos(2.0 * np.pi * x)
            + 0.18 * np.cos(4.0 * np.pi * x)
        )

    for start in starts:
        start = np.clip(start, eps, np.pi - eps)
        result = minimize(
            lambda theta: -value(theta),
            start,
            jac=objective_gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=(
                {
                    **constraint,
                    "jac": lambda theta: -np.sin(theta) / n,
                },
            ),
            options={"maxiter": 300, "ftol": 2e-10},
        )
        candidate = value(result.x)
        if np.isfinite(candidate) and abs(density_constraint(result.x)) < 1e-6:
            candidates.append(candidate)

    return float(max(candidates))


@lru_cache(maxsize=512)
def _unit_scgf(field, bias):
    """SCGF with D=1; use the flat branch unless it is linearly unstable."""
    centered_bias = bias + field
    instability = field * field - centered_bias * centered_bias
    # The first Fourier mode becomes unstable at wave number 2*pi.
    if instability > (2.0 * np.pi) ** 2:
        return _traveling_profile_value(field, bias)
    return bias * (bias + 2.0 * field) / 4.0


def predict_at(experiments, diffusivity):
    """Return the leading weak-asymmetry SCGF for each experiment.

    The uniform-profile branch is ``D * bias * (bias + 2 * field) / 4`` at
    half filling. At strong fields and opposing biases, a traveling density
    profile can dominate; that branch is obtained from the MFT variational
    problem in :func:`_traveling_profile_value`.
    """
    field = np.asarray([e["field"] for e in experiments], dtype=float)
    bias = np.asarray([e["bias"] for e in experiments], dtype=float)
    values = np.empty(field.shape, dtype=float)
    for index, (e, lam) in enumerate(zip(field, bias)):
        values[index] = diffusivity * _unit_scgf(float(e), float(lam))
    return values


class Model:
    def __init__(self):
        self.diffusivity = 1.

    def fit(self, records):
        """Fit the single rate prefactor using the known measurement errors."""
        inputs = [record["input"] for record in records]
        targets = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)

        # The prediction is linear in D: y_i = D * x_i.  Inverse-variance
        # weighted least squares is the Gaussian maximum-likelihood estimate.
        x = predict_at(inputs, 1.0)
        weights = 1.0 / np.square(sigma)
        denominator = np.sum(weights * np.square(x))
        if denominator > 0 and np.isfinite(denominator):
            estimate = np.sum(weights * x * targets) / denominator
            if np.isfinite(estimate):
                self.diffusivity = float(np.clip(estimate, 0.8, 1.2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
