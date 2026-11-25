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
    w, d, angle = (experiment[key] for key in ('frequency', 'thickness', 'angle'))
    lower, upper = np.asarray(experiment['window'])*d
    z = (lower+upper)/2+(upper-lower)*NODES/2
    coefficients = amplitudes(w, d, angle, plasma_frequency)
    electric, electric_z, current, current_z = (
        mode_matrix(z, w, d, angle, plasma_frequency) @ coefficients).T
    k = w*np.sin(angle)
    # The time-averaged mechanical heating, after cancelling the common
    # factor of one half between the heat and incident-wave power, is
    #
    #   gamma |J|^2 + nu (|d_z J|^2 + k^2 |J|^2)
    #
    # divided by p^2.  It is important to use this local form rather than
    # replacing the viscous term with -Re(conj(J) Laplacian(J)): those two
    # expressions differ by a boundary flux when the calorimeter covers
    # only part of the slab.
    heat = GAMMA*abs(current)**2 + VISCOSITY*(
        abs(current_z)**2 + k*k*abs(current)**2)
    return float((upper-lower)/2*np.dot(WEIGHTS, heat)/(plasma_frequency**2*np.cos(angle)))


class Model:
    def __init__(self):
        self.plasma_frequency = 1.0

    def fit(self, records):
        """Fit the single plasma-frequency parameter by weighted least squares."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record['input'] for record in records]
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)
        if (not np.isfinite(values).all() or not np.isfinite(sigmas).all()
                or np.any(sigmas <= 0)):
            raise ValueError("calibration values and sigmas must be finite, with positive sigmas")

        lower, upper = .85, 1.15

        def objective(p):
            prediction = np.asarray(
                [response(experiment, p) for experiment in experiments],
                dtype=float)
            residual = (prediction-values)/sigmas
            return float(residual @ residual)

        # A coarse scan makes the fit robust to the mild resonant structure
        # of the slab response.  Refine the best grid cell with a bounded
        # golden-section search, avoiding a dependency on scipy.
        grid = np.linspace(lower, upper, 101)
        losses = np.asarray([objective(p) for p in grid])
        best = int(np.argmin(losses))
        if best == 0 or best == len(grid)-1:
            fitted = float(grid[best])
        else:
            a, b = float(grid[best-1]), float(grid[best+1])
            golden = (np.sqrt(5.0)-1.0)/2.0
            x1, x2 = b-golden*(b-a), a+golden*(b-a)
            f1, f2 = objective(x1), objective(x2)
            for _ in range(60):
                if f1 <= f2:
                    b, x2, f2 = x2, x1, f1
                    x1 = b-golden*(b-a)
                    f1 = objective(x1)
                else:
                    a, x1, f1 = x1, x2, f2
                    x2 = a+golden*(b-a)
                    f2 = objective(x2)
            fitted = (a+b)/2.0

        self.plasma_frequency = float(np.clip(fitted, lower, upper))
        return self

    def predict(self, experiments):
        return np.asarray([response(e, self.plasma_frequency) for e in experiments], dtype=float)
