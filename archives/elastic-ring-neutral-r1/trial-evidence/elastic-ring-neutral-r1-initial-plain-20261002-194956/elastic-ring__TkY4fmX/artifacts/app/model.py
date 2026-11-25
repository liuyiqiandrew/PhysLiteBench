"""Model for the planar elastic ring."""

import numpy as np
from numpy.polynomial.legendre import leggauss


# The documented controls are benign for this quadrature, including the
# integrable endpoint singularity in the constrained configurational measure.
NODES, WEIGHTS = leggauss(128)


def shape_measure(angle):
    """Induced Cartesian surface element of the fixed-side rhombus manifold."""
    u = np.array([1.0, 0.0])
    v = np.array([np.cos(angle), np.sin(angle)])
    positions = np.array([-u-v, u-v, u+v, -u+v]) / 2.0
    orientation = np.column_stack([-positions[:, 1], positions[:, 0]]).ravel()
    dv = np.array([-np.sin(angle), np.cos(angle)])
    internal = np.array([-dv, -dv, dv, dv]).ravel() / 2.0
    tangent = np.column_stack([orientation, internal])
    return float(np.sqrt(np.linalg.det(tangent.T @ tangent)))


def _constraint_jacobian_factor(angle):
    """Coarea factor from the four side-length constraints.

    At a unit-side rhombus, the Gram determinant of their gradients is
    ``16*sin(angle)**2``.  Integrating the transverse Gaussian spring modes
    therefore contributes the inverse square root of this determinant.
    """
    return 1.0 / (4.0 * np.sin(angle))


def _angular_expectation(stiffness, temperature, cutoff, preferred, observable):
    half_width = (np.pi - 2.0 * cutoff) / 2.0
    angles = cutoff + (NODES + 1.0) * half_width
    exponent = -stiffness * (1.0 - np.cos(angles - preferred)) / temperature
    # Subtracting the maximum avoids needless underflow for wider inputs.
    exponent -= np.max(exponent)
    measure = np.array([shape_measure(angle) for angle in angles])
    measure *= _constraint_jacobian_factor(angles)
    weights = WEIGHTS * measure * np.exp(exponent)
    return float(weights @ observable(angles) / weights.sum())


def predict_at(experiments, stiffness):
    values = []
    for experiment in experiments:
        readout = experiment['readout']
        preferred = float(experiment['preferred'])
        if readout == 'torque':
            values.append(-stiffness*np.sin(float(experiment['angle'])-preferred))
        elif readout == 'sine':
            values.append(_angular_expectation(
                stiffness, float(experiment['temperature']),
                float(experiment['cutoff']), preferred, np.sin))
        elif readout == 'cosine2':
            values.append(_angular_expectation(
                stiffness, float(experiment['temperature']),
                float(experiment['cutoff']), preferred,
                lambda angle: np.cos(2.0*angle)))
        else:
            raise ValueError(f"unknown readout: {readout!r}")
    return np.asarray(values, dtype=float)


def _weighted_loss(stiffness, inputs, values, sigmas):
    residual = (predict_at(inputs, stiffness) - values) / sigmas
    return float(residual @ residual)


class Model:
    def __init__(self):
        self.stiffness = None

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        inputs = [record['input'] for record in records]
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)
        if (not np.isfinite(values).all() or not np.isfinite(sigmas).all()
                or np.any(sigmas <= 0)):
            raise ValueError("calibration values and sigmas must be finite, positive numbers")

        # The supplied calibration is torque-only, for which the weighted
        # least-squares estimate is linear and exact.
        if all(experiment['readout'] == 'torque' for experiment in inputs):
            design = np.asarray([
                -np.sin(float(experiment['angle']) - float(experiment['preferred']))
                for experiment in inputs
            ])
            weights = 1.0 / sigmas**2
            denominator = float(weights @ (design * design))
            estimate = float((weights * design) @ values / denominator)
            self.stiffness = float(np.clip(estimate, 0.5, 1.5))
            return self

        # One-dimensional bounded least squares for any mixed calibration.
        left, right = 0.5, 1.5
        golden = (np.sqrt(5.0) - 1.0) / 2.0
        x1 = right - golden * (right-left)
        x2 = left + golden * (right-left)
        f1 = _weighted_loss(x1, inputs, values, sigmas)
        f2 = _weighted_loss(x2, inputs, values, sigmas)
        for _ in range(80):
            if f1 <= f2:
                right, x2, f2 = x2, x1, f1
                x1 = right - golden * (right-left)
                f1 = _weighted_loss(x1, inputs, values, sigmas)
            else:
                left, x1, f1 = x1, x2, f2
                x2 = left + golden * (right-left)
                f2 = _weighted_loss(x2, inputs, values, sigmas)
        candidates = [0.5, 1.5, (left+right)/2.0]
        self.stiffness = float(min(
            candidates, key=lambda k: _weighted_loss(k, inputs, values, sigmas)))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.stiffness)
