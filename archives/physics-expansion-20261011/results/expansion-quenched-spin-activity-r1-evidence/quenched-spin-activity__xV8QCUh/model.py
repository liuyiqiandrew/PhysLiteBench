"""Model for the switching activity of the two-population Curie--Weiss model.

The long-time scaled cumulant generating function is the largest eigenvalue
of the count-tilted Markov generator.  Detailed balance lets us symmetrize
that generator.  In the N -> infinity limit its largest eigenvalue is the
maximum of a two-angle mean-field variational problem; see
``_scaled_response`` below.
"""

import numpy as np
from scipy.optimize import minimize


_LOWER_ATTEMPT_RATE = 0.8
_UPPER_ATTEMPT_RATE = 1.4
_ANGLE_BOUND = np.pi / 2


def _parameters(experiment):
    """Return the numeric parameters in the order used internally."""
    return (
        float(experiment["coupling"]),
        float(experiment["field"]),
        float(experiment["disorder"]),
        float(experiment["bias_plus"]),
        float(experiment["bias_minus"]),
    )


def _independent_response(field, disorder, bias_plus, bias_minus):
    """Exact scaled SCGF when the coupling is zero."""
    local_fields = np.array([field + disorder, field - disorder], dtype=float)
    biases = np.array([bias_plus, bias_minus], dtype=float)
    tanh_fields = np.tanh(local_fields)
    inverse_cosh_squared = 1.0 / np.cosh(local_fields) ** 2
    terms = np.sqrt(tanh_fields**2 + np.exp(-2.0 * biases) * inverse_cosh_squared)
    return float(0.25 * np.sum(terms - 1.0))


def _variational_value(angles, coupling, field, disorder, biases):
    """Scaled variational objective, with attempt_rate factored out."""
    magnetizations = np.sin(angles)
    total_magnetization = float(np.mean(magnetizations))
    local_fields = coupling * total_magnetization + np.array(
        [field + disorder, field - disorder], dtype=float
    )
    tanh_fields = np.tanh(local_fields)
    transverse = np.exp(-biases) / np.cosh(local_fields)
    return float(
        0.25
        * (
            np.sum(magnetizations * tanh_fields + transverse * np.cos(angles))
            - 2.0
        )
    )


def _variational_gradient(angles, coupling, field, disorder, biases):
    """Analytic gradient of ``_variational_value``."""
    magnetizations = np.sin(angles)
    cos_angles = np.cos(angles)
    total_magnetization = float(np.mean(magnetizations))
    local_fields = coupling * total_magnetization + np.array(
        [field + disorder, field - disorder], dtype=float
    )
    tanh_fields = np.tanh(local_fields)
    sech_squared = 1.0 / np.cosh(local_fields) ** 2
    transverse = np.exp(-biases) / np.cosh(local_fields)

    shared = coupling * np.sum(
        magnetizations * sech_squared - transverse * np.cos(angles) * tanh_fields
    )
    gradient = cos_angles * (tanh_fields + 0.5 * shared) - transverse * magnetizations
    return 0.25 * gradient


def _scaled_response(experiment):
    """Return psi/attempt_rate for one experiment."""
    coupling, field, disorder, bias_plus, bias_minus = _parameters(experiment)
    biases = np.array([bias_plus, bias_minus], dtype=float)

    if bias_plus == 0.0 and bias_minus == 0.0:
        return 0.0
    if coupling == 0.0:
        return _independent_response(field, disorder, bias_plus, bias_minus)

    objective_args = (coupling, field, disorder, biases)

    def objective(angles):
        return -_variational_value(angles, *objective_args)

    def jacobian(angles):
        return -_variational_gradient(angles, *objective_args)

    # Both sides and the center cover the usual pair of mean-field maxima and
    # the asymmetric cases produced by the two different local fields.
    starts = (-1.5, 0.0, 1.5)
    best = -np.inf
    for first in starts:
        for second in starts:
            result = minimize(
                objective,
                np.array([first, second], dtype=float),
                jac=jacobian,
                method="L-BFGS-B",
                bounds=[(-_ANGLE_BOUND, _ANGLE_BOUND)] * 2,
                options={"ftol": 1e-14, "gtol": 1e-11, "maxiter": 300},
            )
            if result.success or np.isfinite(result.fun):
                best = max(best, -float(result.fun))

    if not np.isfinite(best):
        raise RuntimeError("Unable to optimize the tilted-generator response")
    return float(best)


def response(e, kappa):
    """Return the predicted readout for one experiment."""
    return float(kappa) * _scaled_response(e)


def predict_at(experiments, attempt_rate):
    """Predict in input order and always return a one-dimensional ndarray."""
    return np.asarray(
        [response(experiment, float(attempt_rate)) for experiment in experiments],
        dtype=float,
    )


class Model:
    def __init__(self):
        self.attempt_rate = None

    def fit(self, records):
        """Fit the common attempt rate by the stated Gaussian likelihood."""
        records = list(records)
        if not records:
            raise ValueError("fit requires at least one calibration record")

        design = np.asarray([_scaled_response(r["input"]) for r in records], dtype=float)
        values = np.asarray([float(r["value"]) for r in records], dtype=float)
        sigma = np.asarray([float(r["sigma"]) for r in records], dtype=float)
        if not np.isfinite(design).all() or not np.isfinite(values).all():
            raise ValueError("calibration records must contain finite values")
        if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise ValueError("calibration sigma values must be positive and finite")

        weights = 1.0 / sigma**2
        denominator = float(np.sum(weights * design**2))
        if denominator == 0.0:
            raise ValueError("calibration records do not identify attempt_rate")
        estimate = float(np.sum(weights * design * values) / denominator)
        self.attempt_rate = float(
            np.clip(estimate, _LOWER_ATTEMPT_RATE, _UPPER_ATTEMPT_RATE)
        )
        return self

    def predict(self, experiments):
        if self.attempt_rate is None:
            raise RuntimeError("Model must be fit before prediction")
        return predict_at(experiments, self.attempt_rate)
