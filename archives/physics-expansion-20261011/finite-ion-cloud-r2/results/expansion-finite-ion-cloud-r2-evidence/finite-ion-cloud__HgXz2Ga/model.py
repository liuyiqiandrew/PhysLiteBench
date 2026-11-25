"""Finite-cell osmotic coefficient for the two-dimensional ionic cell."""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import hyp2f1


def _one_ion_coefficient(gamma, length):
    """Return phi for one ion, using the radial partition integral."""
    slope = 2.0 - gamma
    if abs(slope) < 1.0e-8:
        radial_log_derivative = 1.0 / length
    else:
        radial_log_derivative = slope / (-np.expm1(-slope * length))
    return -np.expm1(-2.0 * length) * radial_log_derivative / 2.0


def _angular_pair_factor(separation, gamma):
    """Angular average of the two-ion distance at a fixed radial ratio."""
    z = np.exp(-2.0 * separation)
    return hyp2f1(-0.5 * gamma, -0.5 * gamma, 1.0, z)


def _two_ion_coefficient(gamma, length):
    """Return phi from the exact two-ion finite-cell partition function."""
    # Order the logarithmic radii as t1 >= t2.  The radial powers in this
    # ordered sector are exp(s1*t1 + s2*t2), with the pair distance factored
    # at the larger radius.  The other sector supplies the factor two.
    s1 = 2.0 - gamma
    s2 = 2.0 - 2.0 * gamma
    combined = s1 + s2

    def ordered_integrand(separation):
        remaining = length - separation
        if abs(combined) < 1.0e-8:
            inner = remaining
        else:
            inner = np.expm1(combined * remaining) / combined
        return (
            _angular_pair_factor(separation, gamma)
            * np.exp(s1 * separation)
            * inner
        )

    # The dimensionless configurational integral, up to constants independent
    # of the outer radius, and its derivative with respect to length.
    partition = 2.0 * quad(
        ordered_integrand, 0.0, length, epsabs=2.0e-11, epsrel=2.0e-11,
        limit=150,
    )[0]
    boundary_integral = quad(
        lambda separation: _angular_pair_factor(separation, gamma)
        * np.exp(-s2 * separation),
        0.0,
        length,
        epsabs=2.0e-11,
        epsrel=2.0e-11,
        limit=150,
    )[0]
    partition_derivative = 2.0 * np.exp(combined * length) * boundary_integral

    # A = pi*a^2*(exp(2*length)-1), so
    # phi = (1-exp(-2*length))/(2*N) * d(log Z)/d(length).
    return -np.expm1(-2.0 * length) * partition_derivative / (4.0 * partition)


def coefficient(gamma, count, ratio):
    """Finite-cell osmotic coefficient for one or two mobile ions."""
    gamma = float(gamma)
    count = int(count)
    length = np.log(float(ratio))
    if count == 1:
        return _one_ion_coefficient(gamma, length)
    if count == 2:
        return _two_ion_coefficient(gamma, length)
    raise ValueError("ion_count must be 1 or 2")


def predict_at(experiments,coupling):
    if coupling is None:
        raise ValueError("fit the model before predicting")
    return np.asarray([
        coefficient(
            e['temperature_multiplier'] * coupling,
            e['ion_count'],
            e['radius_ratio'],
        )
        for e in experiments
    ], dtype=float)


class Model:
    def __init__(self):
        self.coupling = None

    def fit(self,records):
        """Fit the shared reference coupling by weighted least squares."""
        records = list(records)
        inputs = [record['input'] for record in records]
        measured = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)

        # Calibration is explicitly made with one-ion preparations.
        if not inputs:
            raise ValueError('at least one calibration record is required')
        if any(int(item['ion_count']) != 1 for item in inputs):
            raise ValueError('calibration records must contain one ion')
        if (np.any(~np.isfinite(measured)) or np.any(~np.isfinite(sigma))
                or np.any(sigma <= 0.0)):
            raise ValueError('calibration values and sigma must be finite, with sigma > 0')

        def objective(coupling):
            predicted = predict_at(inputs, coupling)
            residual = (predicted - measured) / sigma
            return float(np.dot(residual, residual))

        result = minimize_scalar(
            objective,
            bounds=(0.7, 1.4),
            method='bounded',
            options={'xatol': 1.0e-12},
        )
        if not result.success:
            raise RuntimeError('coupling fit did not converge')
        self.coupling = float(result.x)
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.coupling)
