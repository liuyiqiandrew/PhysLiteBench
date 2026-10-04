import numpy as np
from numpy.polynomial.legendre import leggauss

GAMMA = .06
VISCOSITY = .04
NODES, WEIGHTS = leggauss(64)
PLASMA_BOUNDS = (.85, 1.15)


def mode_matrix(z, frequency, thickness, angle, plasma_frequency):
    w = frequency
    k = w*np.sin(angle)
    roots = np.roots([VISCOSITY, GAMMA-1j*w-VISCOSITY*w*w,
                      -w*w*(GAMMA-1j*w)-1j*w*plasma_frequency**2])
    vertical = np.sqrt(roots-k*k+0j)
    z = np.atleast_1d(z)
    matrix = np.zeros((len(z), 4, 4), dtype=complex)
    for j in range(2):
        current = (roots[j]-w*w)/(1j*w)
        for side in range(2):
            sign = 1 if side == 0 else -1
            anchor = 0 if side == 0 else thickness
            derivative = 1j*sign*vertical[j]
            wave = np.exp(derivative*(z-anchor))
            matrix[:, :, 2*j+side] = wave[:, None]*np.array(
                [1, derivative, current, derivative*current])
    return matrix


def amplitudes(frequency, thickness, angle, plasma_frequency):
    left, right = mode_matrix([0, thickness], frequency, thickness, angle, plasma_frequency)
    normal = frequency*np.cos(angle)
    boundary = np.array([left[1]+1j*normal*left[0],
                         right[1]-1j*normal*right[0], left[2], right[2]])
    return np.linalg.solve(boundary, [2j*normal, 0, 0, 0])


def response(experiment, plasma_frequency):
    w, d, angle = (experiment[key] for key in ('frequency', 'thickness', 'angle'))
    lower, upper = np.asarray(experiment['window'])*d
    z = (lower+upper)/2+(upper-lower)*NODES/2
    coefficients = amplitudes(w, d, angle, plasma_frequency)
    electric, electric_z, current, current_z = (
        mode_matrix(z, w, d, angle, plasma_frequency) @ coefficients).T
    k = w*np.sin(angle)
    current_zz = ((GAMMA-1j*w+VISCOSITY*k*k)*current-plasma_frequency**2*electric)/VISCOSITY
    heat = GAMMA*abs(current)**2-VISCOSITY*np.real(
        np.conj(current)*(current_zz-k*k*current))
    return float((upper-lower)/2*np.dot(WEIGHTS, heat)/(plasma_frequency**2*np.cos(angle)))


class Model:
    def __init__(self):
        self.plasma_frequency = 1.0

    def fit(self, records):
        """Calibrate the common plasma frequency from weighted observations.

        The detector uncertainty is supplied with each calibration record, so
        the likelihood (up to an additive constant) is the weighted residual
        sum of squares.  There is only one unknown parameter; a bounded
        one-dimensional search is both more robust and less dependent on
        optional numerical packages than a general-purpose optimizer.
        """
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record['input'] for record in records]
        observed = np.asarray([record['value'] for record in records],
                              dtype=float)
        sigma = np.asarray([record['sigma'] for record in records],
                           dtype=float)
        if (not np.isfinite(observed).all() or
                not np.isfinite(sigma).all() or np.any(sigma <= 0)):
            raise ValueError("calibration values and sigma must be finite, with sigma > 0")

        def objective(plasma_frequency):
            predicted = np.asarray(
                [response(experiment, plasma_frequency)
                 for experiment in experiments],
                dtype=float,
            )
            if not np.isfinite(predicted).all():
                return np.inf
            residual = (predicted - observed) / sigma
            return float(residual @ residual)

        lo, hi = PLASMA_BOUNDS

        # Locate a bracket for the best basin first.  This also makes the
        # calibration behave sensibly if a future data set is not perfectly
        # quadratic over the allowed interval.
        grid = np.linspace(lo, hi, 33)
        scores = np.asarray([objective(value) for value in grid])
        best = int(np.argmin(scores))
        if best == 0 or best == len(grid) - 1:
            fitted = float(grid[best])
        else:
            left, right = float(grid[best - 1]), float(grid[best + 1])
            golden = (np.sqrt(5.0) - 1.0) / 2.0
            x1 = right - golden * (right - left)
            x2 = left + golden * (right - left)
            f1, f2 = objective(x1), objective(x2)
            for _ in range(60):
                if f1 <= f2:
                    right, x2, f2 = x2, x1, f1
                    x1 = right - golden * (right - left)
                    f1 = objective(x1)
                else:
                    left, x1, f1 = x1, x2, f2
                    x2 = left + golden * (right - left)
                    f2 = objective(x2)
            fitted = (left + right) / 2.0

        self.plasma_frequency = float(np.clip(fitted, lo, hi))
        return self

    def predict(self, experiments):
        return np.asarray([response(e, self.plasma_frequency) for e in experiments], dtype=float)
