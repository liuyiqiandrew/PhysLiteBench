import numpy as np


_SIX_PI = 6.0 * np.pi
_RESPONSE_BOUNDS = (0.6, 1.6)


def field_data(experiment):
    """Return the incident E, H, and spatial derivative of E at the particle.

    ``derivative[i, j]`` is the derivative of the j-th electric-field
    component with respect to the i-th coordinate.  Keeping the complex
    amplitudes until the force is evaluated is important for coherent waves.
    """
    electric = np.zeros(3, dtype=complex)
    magnetic = np.zeros(3, dtype=complex)
    derivative = np.zeros((3, 3), dtype=complex)
    position = np.asarray(experiment['position'], dtype=float)
    for wave in experiment['waves']:
        direction = np.asarray(wave['direction'], dtype=float)
        amplitude = (np.asarray(wave['real'], dtype=float)
                     + 1j * np.asarray(wave['imag'], dtype=float))
        local = amplitude*np.exp(1j*direction@position)
        electric += local
        magnetic += np.cross(direction, local)
        derivative += 1j*np.outer(direction, local)
    return electric, magnetic, derivative


def coefficients(experiment):
    """Return the two force coefficients multiplying Re(alpha), Im(alpha).

    For a point electric dipole, the time-averaged force due to the incident
    field can be written

        F_i = 1/2 Re[alpha * E_j * d_i(conj(E_j))].

    With A_i = d_i(E_j) conj(E_j), this is

        F_i = Re(alpha) Re(A_i)/2 + Im(alpha) Im(A_i)/2.

    This is equivalent to evaluating the Lorentz force including the magnetic
    term, but is considerably less error-prone for coherent superpositions of
    waves.  In particular, the imaginary coefficient is the canonical
    momentum density; replacing it by the Poynting vector is not generally
    valid for an arbitrary two-wave field.
    """
    electric, _, derivative = field_data(experiment)
    axis = np.asarray(experiment['axis'], dtype=float)
    local_product = derivative @ electric.conj()
    return np.array([
        0.5 * axis @ local_product.real,
        0.5 * axis @ local_product.imag,
    ])


def polarizability(response_strength):
    response_strength = float(response_strength)
    return response_strength/(1-1j*response_strength/_SIX_PI)


def predict_at(experiments, response_strength):
    if not np.isfinite(response_strength):
        raise ValueError("response_strength must be finite")
    alpha = polarizability(response_strength)
    result = np.asarray([
        coefficients(experiment) @ np.array([alpha.real, alpha.imag])
        for experiment in experiments
    ], dtype=float)
    return result.reshape(-1)


def _objective(response_strength, design, values, weights):
    """Weighted sum of squared residuals for one response strength."""
    alpha = polarizability(response_strength)
    prediction = design @ np.array([alpha.real, alpha.imag])
    residual = (prediction - values) * weights
    return float(residual @ residual)


def _fit_response_strength(design, values, sigmas):
    """Minimize the one-dimensional weighted least-squares objective.

    The objective is smooth on the documented interval.  A coarse scan makes
    the procedure robust to an unusual calibration set, and golden-section
    refinement gives more than enough precision for the supplied measurement
    noise without adding a SciPy dependency.
    """
    if design.shape[0] == 0:
        raise ValueError("at least one calibration record is required")
    if not (np.isfinite(values).all() and np.isfinite(sigmas).all()):
        raise ValueError("calibration values and sigmas must be finite")
    if np.any(sigmas <= 0):
        raise ValueError("calibration sigmas must be positive")

    weights = 1.0 / sigmas
    lower, upper = _RESPONSE_BOUNDS
    grid = np.linspace(lower, upper, 257)
    objective_values = np.array([
        _objective(s, design, values, weights) for s in grid
    ])
    best = int(np.argmin(objective_values))
    if best == 0 or best == len(grid) - 1:
        return float(grid[best])

    left, right = grid[best - 1], grid[best + 1]
    golden_ratio = (np.sqrt(5.0) - 1.0) / 2.0
    x1 = right - golden_ratio * (right - left)
    x2 = left + golden_ratio * (right - left)
    f1 = _objective(x1, design, values, weights)
    f2 = _objective(x2, design, values, weights)
    for _ in range(100):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - golden_ratio * (right - left)
            f1 = _objective(x1, design, values, weights)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + golden_ratio * (right - left)
            f2 = _objective(x2, design, values, weights)
    return float((left + right) / 2.0)


class Model:
    def __init__(self):
        self.response_strength = None

    def fit(self, records):
        records = list(records)
        design = np.asarray([coefficients(record['input']) for record in records],
                            dtype=float)
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)
        self.response_strength = _fit_response_strength(design, values, sigmas)
        return self

    def predict(self, experiments):
        if self.response_strength is None:
            raise ValueError("fit the model before calling predict")
        return predict_at(experiments, self.response_strength)
