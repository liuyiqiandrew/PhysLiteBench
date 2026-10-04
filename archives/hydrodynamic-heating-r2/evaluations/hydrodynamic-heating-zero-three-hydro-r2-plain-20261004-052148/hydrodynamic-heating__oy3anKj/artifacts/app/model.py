import numpy as np
from numpy.polynomial.legendre import leggauss

GAMMA = .06
VISCOSITY = .04
NODES, WEIGHTS = leggauss(64)


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
    """Return the normalized heat collected from one depth window.

    The phasors below are normalized to unit incident electric-field
    amplitude.  Since ``J = n0*q*v`` and ``p**2 = n0*q**2/m``, the
    time-averaged heat generated per unit volume, with the common factor of
    one half omitted, is

        (gamma*|J|**2 + nu*(|dJ/dz|**2 + k**2*|J|**2)) / p**2.

    The same factor of one half occurs in the incident Poynting flux, so it
    cancels in the normalized detector readout.  It is important to use the
    local gradient form here: integrating by parts would only be equivalent
    for a window whose endpoints are both no-slip walls.
    """
    w, d, angle = (experiment[key] for key in ('frequency', 'thickness', 'angle'))
    lower, upper = np.asarray(experiment['window'])*d
    z = (lower+upper)/2+(upper-lower)*NODES/2
    coefficients = amplitudes(w, d, angle, plasma_frequency)
    electric, electric_z, current, current_z = (
        mode_matrix(z, w, d, angle, plasma_frequency) @ coefficients).T
    k = w*np.sin(angle)
    heat = GAMMA*abs(current)**2 + VISCOSITY*(
        abs(current_z)**2 + k*k*abs(current)**2)
    value = (upper-lower)/2*np.dot(WEIGHTS, heat)
    value /= plasma_frequency**2*np.cos(angle)
    return float(np.real(value))


class Model:
    def __init__(self):
        self.plasma_frequency = 1.0

    def fit(self, records):
        """Fit the common plasma frequency by weighted least squares.

        The calibration uncertainty is fixed and independent between
        records, so maximizing the Gaussian likelihood is equivalent to
        minimizing the sum of squared standardized residuals.  A bounded
        one-dimensional search is sufficient and avoids making the model
        depend on an optional optimization package.
        """
        records = list(records)
        if not records:
            raise ValueError("records must not be empty")

        experiments = [record['input'] for record in records]
        observed = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)
        if (not np.isfinite(observed).all() or
                not np.isfinite(sigma).all() or np.any(sigma <= 0)):
            raise ValueError("calibration values and sigma must be finite, with sigma > 0")

        lo, hi = .85, 1.15

        def objective(p):
            predicted = np.asarray(
                [response(experiment, p) for experiment in experiments],
                dtype=float,
            )
            residual = (predicted-observed)/sigma
            return float(residual @ residual)

        # Locate the best basin first.  The model is smooth in the permitted
        # interval, but the coarse scan also makes the fit robust to any
        # narrow interference features in a user-supplied calibration set.
        grid = np.linspace(lo, hi, 61)
        values = np.asarray([objective(p) for p in grid])
        best = int(np.argmin(values))
        if best == 0 or best == len(grid)-1:
            self.plasma_frequency = float(grid[best])
            return self

        left, right = grid[best-1], grid[best+1]
        # Golden-section minimization on the selected bracket.
        golden = (np.sqrt(5.0)-1.0)/2.0
        x1 = right - golden*(right-left)
        x2 = left + golden*(right-left)
        f1, f2 = objective(x1), objective(x2)
        for _ in range(80):
            if f1 <= f2:
                right, x2, f2 = x2, x1, f1
                x1 = right - golden*(right-left)
                f1 = objective(x1)
            else:
                left, x1, f1 = x1, x2, f2
                x2 = left + golden*(right-left)
                f2 = objective(x2)
        self.plasma_frequency = float((left+right)/2)
        return self

    def predict(self, experiments):
        return np.asarray([response(e, self.plasma_frequency) for e in experiments], dtype=float)
