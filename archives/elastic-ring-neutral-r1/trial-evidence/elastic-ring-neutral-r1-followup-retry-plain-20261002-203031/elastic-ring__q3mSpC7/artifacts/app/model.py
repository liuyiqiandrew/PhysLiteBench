import numpy as np
from numpy.polynomial.legendre import leggauss

NODES, WEIGHTS = leggauss(128)
_LOWER_STIFFNESS = 0.5
_UPPER_STIFFNESS = 1.5


def shape_measure(angle):
    """Induced Cartesian measure of the rhombus shape coordinates."""
    u = np.array([1., 0.])
    v = np.array([np.cos(angle), np.sin(angle)])
    positions = np.array([-u-v, u-v, u+v, -u+v])/2
    orientation = np.column_stack([-positions[:, 1], positions[:, 0]]).ravel()
    dv = np.array([-np.sin(angle), np.cos(angle)])
    internal = np.array([-dv, -dv, dv, dv]).ravel()/2
    tangent = np.column_stack([orientation, internal])
    return np.sqrt(np.linalg.det(tangent.T@tangent))


def _constraint_measure(angle):
    """The nonconstant coarea factor from the four side constraints.

    At a unit-side rhombus, det(Df Df.T) =
    16 * sin(angle)**2 for f_i = side_length_i - 1.  The
    K-to-infinity configurational measure consequently contains the
    reciprocal square root of this determinant.  Its constant factor of four
    cancels from every normalized equilibrium readout.
    """
    return np.sin(angle)


def _equilibrium_measure(angle):
    # The induced measure from ``shape_measure`` is identically one for this
    # rhombus parameterization.  Writing it this way keeps this hot path
    # vectorized over the quadrature nodes.
    induced_measure = np.ones_like(np.asarray(angle, dtype=float))
    return induced_measure / _constraint_measure(angle)


def predict_at(experiments, stiffness):
    """Evaluate the apparatus model for a sequence of experiment inputs."""
    if not np.isfinite(stiffness):
        raise ValueError("stiffness must be finite")

    values = []
    for experiment in experiments:
        preferred = experiment['preferred']
        readout = experiment['readout']
        if readout == 'torque':
            values.append(-stiffness*np.sin(experiment['angle']-preferred))
            continue
        if readout not in ('sine', 'cosine2'):
            raise ValueError(f"unknown readout: {readout!r}")
        cutoff = experiment['cutoff']
        angles = cutoff+(NODES+1)*(np.pi-2*cutoff)/2
        potential = stiffness*(1-np.cos(angles-preferred))
        measure = _equilibrium_measure(angles)
        exponent = -potential/experiment['temperature']
        exponent -= np.max(exponent)
        weight = WEIGHTS*measure*np.exp(exponent)
        observable = np.sin(angles) if readout == 'sine' else np.cos(2*angles)
        values.append(weight@observable/weight.sum())
    return np.array(values)


class Model:
    def __init__(self):
        self.stiffness = None

    def fit(self, records):
        """Fit stiffness by weighted least squares under the stated bounds."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record['input'] for record in records]
        observations = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)
        if (not np.isfinite(observations).all()
                or not np.isfinite(sigmas).all()
                or np.any(sigmas <= 0)):
            raise ValueError("calibration values and sigmas must be finite; sigmas must be positive")

        # Torque is exactly linear in stiffness.  Use the closed form for the
        # supplied calibration, while retaining a general path for mixed
        # torque/equilibrium calibration records.
        if all(experiment['readout'] == 'torque' for experiment in experiments):
            coefficients = np.asarray([
                -np.sin(experiment['angle'] - experiment['preferred'])
                for experiment in experiments
            ])
            weights = 1.0 / sigmas**2
            denominator = np.sum(weights * coefficients**2)
            if denominator == 0:
                raise ValueError("calibration does not constrain stiffness")
            estimate = np.sum(weights * coefficients * observations) / denominator
            self.stiffness = float(np.clip(estimate, _LOWER_STIFFNESS, _UPPER_STIFFNESS))
            return self

        def objective(value):
            residual = (predict_at(experiments, value) - observations) / sigmas
            return float(residual @ residual)

        # This is a one-dimensional bounded problem.  A grid locates the
        # relevant basin and golden-section refinement supplies the estimate.
        grid = np.linspace(_LOWER_STIFFNESS, _UPPER_STIFFNESS, 65)
        scores = np.asarray([objective(value) for value in grid])
        best = int(np.argmin(scores))
        if best == 0 or best == len(grid) - 1:
            self.stiffness = float(grid[best])
            return self

        left, right = grid[best - 1], grid[best + 1]
        golden = (np.sqrt(5.0) - 1.0) / 2.0
        x1 = right - golden * (right - left)
        x2 = left + golden * (right - left)
        f1, f2 = objective(x1), objective(x2)
        for _ in range(80):
            if f1 <= f2:
                right, x2, f2 = x2, x1, f1
                x1 = right - golden * (right - left)
                f1 = objective(x1)
            else:
                left, x1, f1 = x1, x2, f2
                x2 = left + golden * (right - left)
                f2 = objective(x2)
        self.stiffness = float((left + right) / 2.0)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.stiffness)
