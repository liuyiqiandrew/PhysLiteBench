"""SCGF model and calibration interface for the weakly driven exclusion ring."""
import numpy as np


# Gauss-Legendre nodes are used after removing the square-root endpoint
# singularity from the quadratures defining the periodic profile.
_quad_x, _quad_w = np.polynomial.legendre.leggauss(128)
_quad_u = (_quad_x + 1.0) * (np.pi / 4.0)
_quad_wu = _quad_w * (np.pi / 4.0)
_PI2 = np.pi * np.pi


def _homogeneous_scgf(field, bias):
    """SCGF per diffusivity for a spatially homogeneous profile."""
    return bias * (bias + 2.0 * field) / 4.0


def _profile_integrals(field, amplitude, alpha):
    """Return quarter-period integrals for a modulated density profile."""
    theta = amplitude * np.sin(_quad_u)
    dtheta = amplitude * np.cos(_quad_u)
    c = np.cos(theta)
    ca = np.cos(amplitude)

    # The first-integral expression is non-negative on the physical branch.
    speed2 = (field * field) * (c * c - ca * ca)
    speed2 += alpha * (1.0 / (c * c) - 1.0 / (ca * ca))
    speed = np.sqrt(np.maximum(speed2, 1.0e-300))
    weighted = _quad_wu * dtheta

    quarter = np.sum(weighted / speed)
    inverse_mobility = 4.0 * np.sum(weighted / (c * c * speed))
    second_moment = 4.0 * np.sum(weighted * (np.sin(theta) ** 2) / speed)
    gradient = 4.0 * np.sum(weighted * speed)
    return quarter, inverse_mobility, second_moment, gradient


def _alpha_at_amplitude(field, amplitude):
    """Solve the quarter-period condition for the profile parameter alpha."""
    ca = np.cos(amplitude)
    lo = 0.0
    # Positivity of the first integral requires alpha < E^2 cos(A)^4.
    hi = field * field * ca ** 4 * (1.0 - 1.0e-11)
    for _ in range(58):
        mid = 0.5 * (lo + hi)
        quarter = _profile_integrals(field, amplitude, mid)[0]
        if quarter < 0.25:
            lo = mid
        else:
            hi = mid
    alpha = 0.5 * (lo + hi)
    return alpha, _profile_integrals(field, amplitude, alpha)


def _modulated_value(field, bias):
    """SCGF per diffusivity, including the modulated-profile branch."""
    e2 = field * field
    critical_squared = e2 - 4.0 * _PI2
    target = abs(field + bias)

    # The homogeneous profile is stable unless
    # E^2 - (E+lambda)^2 exceeds 4 pi^2.
    if critical_squared <= 0.0 or target * target >= critical_squared:
        return _homogeneous_scgf(field, bias)

    # Find the largest turning angle for which alpha=0 still has quarter
    # period 1/4.  The quarter period is monotone in the amplitude.
    lo = 1.0e-7
    hi = np.pi / 2.0 - 1.0e-7
    for _ in range(58):
        mid = 0.5 * (lo + hi)
        quarter = _profile_integrals(field, mid, 0.0)[0]
        if quarter < 0.25:
            lo = mid
        else:
            hi = mid
    max_amplitude = 0.5 * (lo + hi)

    if target <= 1.0e-10:
        amplitude = max_amplitude
        alpha = 0.0
        _, inverse_mobility, second_moment, gradient = _profile_integrals(
            field, amplitude, alpha
        )
    else:
        # Along the branch sqrt(alpha)*I decreases from critical to zero.
        lo = 1.0e-7
        hi = max_amplitude
        for _ in range(58):
            mid = 0.5 * (lo + hi)
            alpha, values = _alpha_at_amplitude(field, mid)
            inverse_mobility = values[1]
            branch_target = np.sqrt(max(alpha, 0.0)) * inverse_mobility
            if branch_target > target:
                lo = mid
            else:
                hi = mid
        amplitude = 0.5 * (lo + hi)
        alpha, values = _alpha_at_amplitude(field, amplitude)
        _, inverse_mobility, second_moment, gradient = values

    # For q=2 rho-1, the optimized profile functional is
    # (E+lambda)^2/(4 I) - G/4 - E^2(1-M2)/4.
    return (
        target * target / (4.0 * inverse_mobility)
        - gradient / 4.0
        - e2 * (1.0 - second_moment) / 4.0
    )


def predict_at(experiments, diffusivity):
    """Predict raw SCGF values for field/bias dictionaries."""
    if not experiments:
        return np.empty(0, dtype=float)

    values = []
    for experiment in experiments:
        field = float(experiment["field"])
        bias = float(experiment["bias"])
        values.append(float(diffusivity) * _modulated_value(field, bias))
    return np.asarray(values, dtype=float)


class Model:
    def __init__(self):
        self.diffusivity = 1.0

    def fit(self, records):
        """Fit the common diffusivity by weighted least squares."""
        if records:
            inputs = [record["input"] for record in records]
            basis = np.asarray(
                [_modulated_value(float(x["field"]), float(x["bias"])) for x in inputs],
                dtype=float,
            )
            observed = np.asarray([record["value"] for record in records], dtype=float)
            sigma = np.asarray([record["sigma"] for record in records], dtype=float)
            weights = 1.0 / (sigma * sigma)
            denominator = float(np.sum(weights * basis * basis))
            if denominator > 0.0 and np.isfinite(denominator):
                estimate = float(np.sum(weights * basis * observed) / denominator)
                self.diffusivity = float(np.clip(estimate, 0.8, 1.2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
