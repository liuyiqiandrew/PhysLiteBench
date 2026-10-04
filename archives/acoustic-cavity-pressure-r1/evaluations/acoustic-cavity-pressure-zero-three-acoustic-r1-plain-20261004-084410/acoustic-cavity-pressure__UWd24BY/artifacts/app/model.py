import numpy as np

RHO0 = 1.
BULK = 1.
EXPONENT = 5.


def fields(length, mode, frequency_ratio, drive, viscosity):
    """Return the complex first-order velocity and density amplitudes.

    The spatial dependence is ``u = U sin(k x)`` and ``rho = R cos(k x)``;
    amplitudes use the convention ``Re[A exp(-i omega t)]``.  In particular,
    ``drive`` is a force-per-volume amplitude, so it appears directly on the
    right hand side of the linear momentum equation.
    """
    wavevector = mode*np.pi/length
    frequency = frequency_ratio*np.sqrt(BULK/RHO0)*wavevector
    velocity = drive/(viscosity*wavevector**2+1j*(BULK*wavevector**2/frequency-RHO0*frequency))
    density = -1j*RHO0*wavevector*velocity/frequency
    return velocity, density


def predict_at(experiments, viscosity):
    result = []
    for e in experiments:
        velocity, density = fields(e['length'], e['mode'], e['frequency_ratio'], e['drive'], viscosity)
        if e['readout'] == 'density':
            value = abs(density)
        elif e['readout'] == 'force':
            # At second order, let rho = rho0 + eps*rho1 + eps^2*rho2 and
            # similarly for u.  The mean momentum equation gives
            #
            #   sigma_2 = rho0 <u1^2> + constant,
            #
            # while fixed total mass determines the constant.  Evaluating at
            # the left wall (where <u1^2> vanishes) and using
            # p''(rho0)=4*BULK/RHO0**2 gives the compressive force increase
            #
            #   -<sigma_2> = B*|R|^2/(2*RHO0**2) + RHO0*|U|^2/4.
            #
            # The second term is the contribution of the full normal traction;
            # omitting it treats the transducer as a pressure-only probe.
            value = (BULK*abs(density)**2/(2.*RHO0**2)
                     + RHO0*abs(velocity)**2/4.)
        else:
            raise ValueError("readout must be 'density' or 'force'")
        result.append(float(value))
    return np.asarray(result, dtype=float)


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        """Fit viscosity by weighted least squares over the calibration data."""
        if not records:
            raise ValueError('at least one calibration record is required')

        experiments = [record['input'] for record in records]
        observations = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record.get('sigma', 1.) for record in records], dtype=float)
        if (not np.isfinite(observations).all() or
                not np.isfinite(sigmas).all() or np.any(sigmas <= 0.)):
            raise ValueError('calibration values and sigmas must be finite, positive numbers')

        lower, upper = .08, .18

        def objective(viscosity):
            residual = (predict_at(experiments, viscosity) - observations) / sigmas
            return float(np.dot(residual, residual))

        # This is a one-dimensional fit.  A coarse scan makes the refinement
        # robust to a non-unimodal objective supplied by arbitrary records,
        # while the local golden-section search gives more than enough
        # precision for the instrument uncertainty.
        grid = np.linspace(lower, upper, 1001)
        costs = np.asarray([objective(v) for v in grid])
        best = int(np.argmin(costs))
        if best == 0 or best == len(grid) - 1:
            self.viscosity = float(grid[best])
            return self

        a, b = float(grid[best - 1]), float(grid[best + 1])
        golden = (np.sqrt(5.) - 1.) / 2.
        x1 = b - golden*(b - a)
        x2 = a + golden*(b - a)
        f1, f2 = objective(x1), objective(x2)
        for _ in range(80):
            if f1 > f2:
                a, x1, f1 = x1, x2, f2
                x2 = a + golden*(b - a)
                f2 = objective(x2)
            else:
                b, x2, f2 = x2, x1, f1
                x1 = b - golden*(b - a)
                f1 = objective(x1)
        self.viscosity = float((a + b) / 2.)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.viscosity)
