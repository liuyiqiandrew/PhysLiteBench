import numpy as np


RESPONSE_MIN = 0.6
RESPONSE_MAX = 1.6
RADIATION_CONSTANT = 6.0 * np.pi


def field_data(experiment):
    """Return the incident E, H, and spatial derivative of E at the particle."""
    electric = np.zeros(3, dtype=np.complex128)
    magnetic = np.zeros(3, dtype=np.complex128)
    # derivative[i, j] is d E[j] / d r[i].
    derivative = np.zeros((3, 3), dtype=np.complex128)
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
    """Return force coefficients multiplying Re(alpha) and Im(alpha).

    For a point electric dipole, the time-averaged incident-field force is

        1/2 Re[(p* dot grad) E + i p* cross H],

    with p = alpha E.  The expression in brackets is complex-linear in the
    local field and is the same as derivative @ E.conj() for a source-free
    plane-wave field.  Keeping the Lorentz form here also handles coherent
    superpositions whose local Poynting vector is not the canonical momentum
    density.
    """
    electric, magnetic, derivative = field_data(experiment)
    axis = np.asarray(experiment['axis'], dtype=float)

    # (E* dot grad) E: derivative has spatial index first.
    directional = derivative.T @ electric.conj()
    complex_force = directional + 1j*np.cross(electric.conj(), magnetic)
    projected = axis @ complex_force

    # Re[alpha* (u + i v)] = Re(alpha) u + Im(alpha) v.
    return 0.5*np.array([projected.real, projected.imag])


def polarizability(response_strength):
    return response_strength/(1-1j*response_strength/RADIATION_CONSTANT)


def predict_at(experiments, response_strength):
    alpha = polarizability(response_strength)
    return np.asarray([
        coefficients(experiment) @ np.array([alpha.real, alpha.imag])
        for experiment in experiments
    ], dtype=float)


def _weighted_error(response_strength, design, values, weights):
    alpha = polarizability(response_strength)
    prediction = design @ np.array([alpha.real, alpha.imag])
    residual = (prediction - values) * weights
    return float(residual @ residual)


def _bounded_fit(design, values, sigmas):
    """Minimize the one-dimensional weighted least-squares objective.

    A small grid makes the method robust to a non-unimodal objective; golden
    section refinement then gives more precision than the calibration noise
    requires without depending on scipy.
    """
    weights = 1.0 / sigmas
    grid = np.linspace(RESPONSE_MIN, RESPONSE_MAX, 2049)
    errors = np.array([
        _weighted_error(value, design, values, weights) for value in grid
    ])
    best_index = int(np.argmin(errors))
    best_value = float(grid[best_index])
    best_error = float(errors[best_index])

    if 0 < best_index < len(grid) - 1:
        left = float(grid[best_index - 1])
        right = float(grid[best_index + 1])
        golden = (np.sqrt(5.0) - 1.0) / 2.0
        x1 = right - golden * (right - left)
        x2 = left + golden * (right - left)
        f1 = _weighted_error(x1, design, values, weights)
        f2 = _weighted_error(x2, design, values, weights)
        for _ in range(80):
            if f1 <= f2:
                right, x2, f2 = x2, x1, f1
                x1 = right - golden * (right - left)
                f1 = _weighted_error(x1, design, values, weights)
            else:
                left, x1, f1 = x1, x2, f2
                x2 = left + golden * (right - left)
                f2 = _weighted_error(x2, design, values, weights)
        refined = 0.5 * (left + right)
        refined_error = _weighted_error(refined, design, values, weights)
        if refined_error < best_error:
            best_value = refined

    return best_value


class Model:
    def __init__(self):
        self.response_strength = None

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError("At least one calibration record is required.")

        design = np.asarray([coefficients(record['input']) for record in records])
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)
        if (not np.isfinite(design).all() or not np.isfinite(values).all()
                or not np.isfinite(sigmas).all() or np.any(sigmas <= 0.0)):
            raise ValueError("Calibration records must contain finite values and positive sigmas.")

        self.response_strength = _bounded_fit(design, values, sigmas)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.response_strength)
