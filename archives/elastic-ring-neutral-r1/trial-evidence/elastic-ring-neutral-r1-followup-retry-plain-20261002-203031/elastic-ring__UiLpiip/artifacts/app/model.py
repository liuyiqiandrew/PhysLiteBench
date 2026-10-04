import numpy as np
from numpy.polynomial.legendre import leggauss

NODES, WEIGHTS = leggauss(128)
LOWER_STIFFNESS = 0.5
UPPER_STIFFNESS = 1.5


def shape_measure(angle):
    """Surface measure of the exact-rhombus shape coordinates."""
    u = np.array([1., 0.])
    v = np.array([np.cos(angle), np.sin(angle)])
    positions = np.array([-u-v, u-v, u+v, -u+v])/2
    orientation = np.column_stack([-positions[:, 1], positions[:, 0]]).ravel()
    dv = np.array([-np.sin(angle), np.cos(angle)])
    internal = np.array([-dv, -dv, dv, dv]).ravel()/2
    tangent = np.column_stack([orientation, internal])
    return np.sqrt(np.linalg.det(tangent.T@tangent))


def configuration_measure(angle):
    """Configurational density of the internal angle in the stiff limit.

    Integrating the four Cartesian side-length fluctuations gives a coarea
    factor 1/sin(angle) for a convex unit rhombus.  The surface measure of
    the overall orientation and angle coordinates is angle-independent.
    """
    return 1.0 / np.sin(angle)


def predict_at(experiments, stiffness):
    values = []
    for experiment in experiments:
        preferred = experiment['preferred']
        readout = experiment['readout']
        if readout == 'torque':
            values.append(-stiffness*np.sin(experiment['angle']-preferred))
            continue
        cutoff = experiment['cutoff']
        angles = cutoff+(NODES+1)*(np.pi-2*cutoff)/2
        potential = stiffness*(1-np.cos(angles-preferred))
        exponent = -potential/experiment['temperature']
        exponent -= np.max(exponent)
        measure = configuration_measure(angles)
        weight = WEIGHTS*measure*np.exp(exponent)
        observable = np.sin(angles) if readout == 'sine' else np.cos(2*angles)
        values.append(weight@observable/weight.sum())
    return np.array(values)


class Model:
    def __init__(self):
        self.stiffness = None

    def fit(self, records):
        """Fit stiffness by the supplied-error weighted least-squares rule."""
        records = list(records)
        if not records:
            raise ValueError("records must contain at least one calibration record")

        inputs = [record['input'] for record in records]
        observed = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)
        if (not np.isfinite(observed).all() or
                not np.isfinite(sigma).all() or np.any(sigma <= 0)):
            raise ValueError("calibration values and sigma must be finite, with sigma > 0")

        # Torque is linear in stiffness, so solve the common torque-only case
        # exactly rather than introducing numerical optimization error.
        if all(item['readout'] == 'torque' for item in inputs):
            design = np.asarray([
                -np.sin(item['angle'] - item['preferred']) for item in inputs
            ])
            weights = 1.0/sigma**2
            denominator = np.sum(weights*design**2)
            if denominator == 0:
                raise ValueError("calibration does not identify stiffness")
            estimate = np.sum(weights*design*observed)/denominator
            self.stiffness = float(np.clip(estimate, LOWER_STIFFNESS,
                                           UPPER_STIFFNESS))
            return self

        weights = 1.0/sigma**2

        def objective(value):
            residual = predict_at(inputs, value) - observed
            return float(np.sum(weights*residual*residual))

        # Search the compact documented interval.  The coarse scan protects
        # against an accidentally non-convex mixed-readout objective, while
        # golden-section refinement gives a precise scalar estimate.
        grid = np.linspace(LOWER_STIFFNESS, UPPER_STIFFNESS, 257)
        costs = np.asarray([objective(value) for value in grid])
        best = int(np.argmin(costs))
        if best == 0 or best == len(grid)-1:
            estimate = grid[best]
        else:
            left, right = grid[best-1], grid[best+1]
            golden = (np.sqrt(5.0)-1.0)/2.0
            x1 = right-golden*(right-left)
            x2 = left+golden*(right-left)
            f1, f2 = objective(x1), objective(x2)
            for _ in range(80):
                if f1 <= f2:
                    right, x2, f2 = x2, x1, f1
                    x1 = right-golden*(right-left)
                    f1 = objective(x1)
                else:
                    left, x1, f1 = x1, x2, f2
                    x2 = left+golden*(right-left)
                    f2 = objective(x2)
            estimate = (left+right)/2.0

        self.stiffness = float(np.clip(estimate, LOWER_STIFFNESS,
                                       UPPER_STIFFNESS))
        return self

    def predict(self, experiments):
        if self.stiffness is None:
            raise ValueError("fit must be called before predict")
        return predict_at(experiments, self.stiffness)
