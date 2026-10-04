from functools import lru_cache
import numpy as np

DAMPING = .04
ROTATION = np.kron(np.eye(2), np.array([[0., -1.], [1., 0.]]))
_GYRO_BOUNDS = (0.7, 1.3)


def energy_matrix(kx, ky, gap):
    """Return the quadratic transverse-field matrix for one Fourier mode.

    The state ordering is ``[y_0, z_0, y_1, z_1]``.  The diagonal terms
    contain the applied field, the two easy-axis fields, and exchange.  The
    remaining terms are the thickness-averaged magnetostatic tensor.

    The off-diagonal ``y-z`` terms between films are easy to miss: they are
    present when the wavevector has a component along the dynamic in-plane
    direction (the y direction here).  They are imaginary because the two
    charges are displaced along z while the Fourier mode is complex.
    """
    k = np.hypot(kx, ky)
    if not np.isfinite(k) or k <= 0:
        raise ValueError("wavevector must have nonzero finite magnitude")

    direction = ky/k
    p = -np.expm1(-k)/k
    q = np.exp(-k*gap)*(-np.expm1(-k))**2/(2*k)
    matrix = np.zeros((4, 4), dtype=complex)
    matrix[np.ix_([0, 2], [0, 2])] = direction**2*np.array([[1-p, q], [q, 1-p]])
    matrix[np.ix_([1, 3], [1, 3])] = np.array([[p, -q], [-q, p]])

    # For the lower-to-upper block, the averaged magnetostatic tensor is
    # q * [[direction**2, -1j*direction], [-1j*direction, -1]].
    # The reverse block is its Hermitian conjugate.
    cross = q*np.array([
        [direction**2, -1j*direction],
        [-1j*direction, -1.],
    ])
    matrix[np.ix_([0, 1], [2, 3])] = cross
    matrix[np.ix_([2, 3], [0, 1])] = cross.conj().T

    matrix += np.diag(np.repeat([.45, .70], 2)+.08*k*k)
    return matrix


@lru_cache(maxsize=512)
def mode_basis(kx, ky, gap):
    energy = energy_matrix(kx, ky, gap)
    generator = (ROTATION-DAMPING*np.eye(4))@energy/(1+DAMPING**2)
    rates, vectors = np.linalg.eig(generator)
    return rates, vectors, np.linalg.inv(vectors)


def predict_at(experiments, gyro_rate):
    try:
        gyro_rate = float(gyro_rate)
    except (TypeError, ValueError):
        raise ValueError("gyro_rate must be finite; call fit before predict") from None
    if not np.isfinite(gyro_rate):
        raise ValueError("gyro_rate must be finite; call fit before predict")

    result = []
    for experiment in experiments:
        kx, ky = experiment['wavevector']
        rates, vectors, inverse = mode_basis(kx, ky, experiment['gap'])
        initial = np.zeros(4, dtype=complex)
        initial[[0, 2]] = np.array(experiment['initial_real'])+1j*np.array(experiment['initial_imag'])
        state = vectors@(np.exp(rates*gyro_rate*experiment['time'])*(inverse@initial))
        value = state[2*experiment['layer']+1]
        result.append(value.real if experiment['quadrature']=='real' else value.imag)
    return np.asarray(result, dtype=float)


def _golden_minimize(function, lower, upper, iterations=80):
    """Minimize a scalar function on a bounded interval.

    This small implementation keeps the model dependent only on NumPy.  The
    coarse scan in ``Model.fit`` supplies an interval containing the best
    basin, so the usual unimodal assumption is local rather than global.
    """
    golden = (np.sqrt(5.) - 1.)/2.
    a, b = float(lower), float(upper)
    c = b - golden*(b-a)
    d = a + golden*(b-a)
    fc, fd = function(c), function(d)
    for _ in range(iterations):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - golden*(b-a)
            fc = function(c)
        else:
            a, c, fc = c, d, fd
            d = a + golden*(b-a)
            fd = function(d)
    return c if fc <= fd else d


class Model:
    def __init__(self):
        self.gyro_rate = None

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record['input'] for record in records]
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)
        if (not np.isfinite(values).all() or
                not np.isfinite(sigmas).all() or (sigmas <= 0).any()):
            raise ValueError("calibration values and sigmas must be finite, with sigma > 0")

        inv_sigma = 1./sigmas

        def objective(rate):
            prediction = predict_at(experiments, float(rate))
            residual = (prediction-values)*inv_sigma
            score = residual @ residual
            return float(score) if np.isfinite(score) else np.inf

        # The parameter is known to lie in this interval.  A scan makes the
        # fit reliable even when the time samples produce several oscillatory
        # minima; refine the best sampled basin with a bounded golden search.
        lower, upper = _GYRO_BOUNDS
        grid = np.linspace(lower, upper, 1001)
        scores = np.asarray([objective(rate) for rate in grid])
        best = int(np.argmin(scores))
        if best == 0 or best == len(grid)-1:
            fitted = grid[best]
        else:
            fitted = _golden_minimize(
                objective, grid[best-1], grid[best+1])

        self.gyro_rate = float(np.clip(fitted, lower, upper))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.gyro_rate)
