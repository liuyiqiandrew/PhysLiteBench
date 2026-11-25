from functools import lru_cache
import numpy as np

DAMPING = .04
ROTATION = np.kron(np.eye(2), np.array([[0., -1.], [1., 0.]]))
GYRO_BOUNDS = (0.7, 1.3)


def energy_matrix(kx, ky, gap):
    k = np.hypot(kx, ky)
    direction = ky/k
    p = -np.expm1(-k)/k
    q = np.exp(-k*gap)*(-np.expm1(-k))**2/(2*k)
    matrix = np.zeros((4, 4), dtype=complex)
    matrix[np.ix_([0, 2], [0, 2])] = direction**2*np.array([[1-p, q], [q, 1-p]])
    matrix[np.ix_([1, 3], [1, 3])] = np.array([[p, -q], [-q, p]])
    matrix += np.diag(np.repeat([.45, .70], 2)+.08*k*k)
    return matrix


@lru_cache(maxsize=512)
def mode_basis(kx, ky, gap):
    energy = energy_matrix(kx, ky, gap)
    generator = (ROTATION-DAMPING*np.eye(4))@energy/(1+DAMPING**2)
    rates, vectors = np.linalg.eig(generator)
    return rates, vectors, np.linalg.inv(vectors)


def predict_at(experiments, gyro_rate):
    result = []
    for experiment in experiments:
        kx, ky = experiment['wavevector']
        rates, vectors, inverse = mode_basis(kx, ky, experiment['gap'])
        initial = np.zeros(4, dtype=complex)
        initial[[0, 2]] = np.array(experiment['initial_real'])+1j*np.array(experiment['initial_imag'])
        state = vectors@(np.exp(rates*gyro_rate*experiment['time'])*(inverse@initial))
        value = state[2*experiment['layer']+1]
        result.append(value.real if experiment['quadrature']=='real' else value.imag)
    return np.array(result)


class Model:
    def __init__(self):
        self.gyro_rate = None

    def fit(self, records):
        """Fit the common gyromagnetic rate to calibration measurements.

        There is only one unknown parameter.  The calibration uncertainties are
        independent Gaussian standard deviations, so minimizing the weighted
        sum of squared residuals is the corresponding maximum-likelihood fit.
        A small grid is used to find the relevant basin before the bounded
        golden-section search; this also avoids relying on an optional
        optimization package.
        """
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record["input"] for record in records]
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigmas = np.asarray([record["sigma"] for record in records], dtype=float)
        if (not np.isfinite(values).all() or
                not np.isfinite(sigmas).all() or np.any(sigmas <= 0)):
            raise ValueError("calibration values and sigmas must be finite, with sigma > 0")

        def objective(rate):
            residual = (predict_at(experiments, float(rate)) - values) / sigmas
            return float(residual @ residual)

        lower, upper = GYRO_BOUNDS
        # The calibration objective is smooth but can contain oscillations for
        # long-time measurements.  Locate the best coarse basin first, then
        # refine only that basin with a bounded one-dimensional search.
        grid = np.linspace(lower, upper, 25)
        scores = np.asarray([objective(rate) for rate in grid])
        best = int(np.argmin(scores))
        if best == 0 or best == len(grid) - 1:
            self.gyro_rate = float(grid[best])
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

        candidates = (left, right, x1, x2)
        self.gyro_rate = float(min(candidates, key=objective))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.gyro_rate)
