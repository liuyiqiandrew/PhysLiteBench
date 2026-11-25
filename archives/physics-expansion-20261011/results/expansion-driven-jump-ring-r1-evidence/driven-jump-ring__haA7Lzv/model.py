"""Model for the periodically driven three-state molecular ring."""

import numpy as np
from scipy.linalg import eigvals, expm
from scipy.optimize import minimize_scalar


def generator(stage, kappa, chi):
    """Return the tilted generator for one stage.

    Matrices use column probability vectors: a transition from ``i`` to
    ``j`` is placed in row ``j``, column ``i``.  The diagonal is not tilted,
    since it contains the ordinary escape rates.
    """
    energy = np.asarray(stage['energies'], dtype=float)
    barrier = np.asarray(stage['barriers'], dtype=float)
    affinity = float(stage['affinity'])
    kappa = float(kappa)
    chi = float(chi)

    matrix = np.zeros((3, 3), dtype=float)
    for i in range(3):
        j = (i + 1) % 3
        forward = kappa * np.exp(energy[i] - barrier[i] + affinity / 6.0)
        backward = kappa * np.exp(energy[j] - barrier[i] - affinity / 6.0)
        matrix[j, i] += forward * np.exp(chi / 3.0)
        matrix[i, j] += backward * np.exp(-chi / 3.0)
        matrix[i, i] -= forward
        matrix[j, j] -= backward
    return matrix


def cycle(e, kappa, chi):
    """Return the tilted propagator over one complete period."""
    result = np.eye(3)
    for stage in e['stages']:
        result = expm(float(stage['duration']) * generator(stage, kappa, chi)) @ result
    return result


def _floquet_multiplier(e, kappa, chi):
    """Return the Perron multiplier of the tilted period propagator."""
    if float(chi) == 0.0:
        # The untilted propagator is stochastic, so its multiplier is exactly
        # one.  Avoid returning a tiny roundoff residue in this case.
        return 1.0

    values = eigvals(cycle(e, kappa, chi))
    # Positive stage durations and open bonds make the period propagator
    # positive.  Its Perron root is therefore the eigenvalue of largest
    # modulus, even though the other two eigenvalues may be complex.
    multiplier = float(np.max(np.abs(values)))
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise FloatingPointError('invalid tilted period multiplier')
    return multiplier


def response(e, kappa):
    """Return the asymptotic log moment growth per period."""
    return float(np.log(_floquet_multiplier(e, kappa, e['field'])))


def predict_at(experiments, attempt_rate):
    """Evaluate the documented prediction for a common attempt rate."""
    return np.asarray(
        [response(e, float(attempt_rate)) for e in experiments], dtype=float
    )


class Model:
    def __init__(self):
        self.attempt_rate = None

    def fit(self, records):
        """Fit the common attempt rate by weighted Gaussian least squares."""
        records = list(records)
        if not records:
            raise ValueError('at least one calibration record is required')

        experiments = [record['input'] for record in records]
        observed = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)
        if (
            not np.all(np.isfinite(observed))
            or not np.all(np.isfinite(sigma))
            or np.any(sigma <= 0.0)
            or observed.shape != sigma.shape
        ):
            raise ValueError(
                'calibration values and sigmas must be finite with positive sigmas'
            )

        def objective(attempt_rate):
            residual = (predict_at(experiments, attempt_rate) - observed) / sigma
            return float(np.dot(residual, residual))

        # There is one unknown and its physical interval is specified by the
        # apparatus, so bounded scalar optimization is sufficient.
        result = minimize_scalar(
            objective,
            bounds=(0.8, 1.3),
            method='bounded',
            options={'xatol': 1e-12, 'maxiter': 500},
        )
        if not result.success or not np.isfinite(result.x):
            raise RuntimeError('could not fit the attempt rate')
        self.attempt_rate = float(np.clip(result.x, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.attempt_rate is None:
            raise RuntimeError('fit must be called before predict')
        return predict_at(experiments, self.attempt_rate)
