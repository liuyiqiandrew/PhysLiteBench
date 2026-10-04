import numpy as np

RHO0 = 1.
BULK = 1.
EXPONENT = 5.


def fields(length, mode, frequency_ratio, drive, viscosity):
    """Return the first-order velocity and left-wall density amplitudes.

    The velocity perturbation is ``V sin(k x)``.  The continuity equation
    then makes the density perturbation ``R cos(k x)``, where ``R`` is the
    value at the left wall.  Complex amplitudes use the convention from the
    README (the physical field is the real part times ``exp(-i omega t)``).
    """
    wavevector = mode*np.pi/length
    frequency = frequency_ratio*np.sqrt(BULK/RHO0)*wavevector
    velocity = drive/(viscosity*wavevector**2+1j*(BULK*wavevector**2/frequency-RHO0*frequency))
    density = -1j*RHO0*wavevector*velocity/frequency
    return velocity, density


def predict_at(experiments, viscosity):
    result = []
    for e in experiments:
        length = e['length']
        mode = e['mode']
        frequency_ratio = e['frequency_ratio']
        drive = e['drive']
        velocity, density = fields(length, mode, frequency_ratio, drive, viscosity)

        if e['readout'] == 'density':
            value = abs(density)
        elif e['readout'] == 'force':
            # Average the conservative momentum flux over the cavity.  At
            # second order this is the wall traction because u=0 at both
            # walls.  The kinetic and equation-of-state contributions are
            # respectively rho0 |V|^2/4 and p'' |R|^2/8.
            curvature = (EXPONENT-1)*BULK/RHO0**2
            value = RHO0*abs(velocity)**2/4 + curvature*abs(density)**2/8
        else:
            raise ValueError(f"unknown readout {e['readout']!r}")
        result.append(float(value))
    return np.asarray(result, dtype=float)


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        """Fit viscosity by weighted least squares against calibration data."""
        records = list(records)
        if not records:
            raise ValueError('at least one calibration record is required')

        experiments = [record['input'] for record in records]
        observations = np.asarray([record['value'] for record in records], dtype=float)
        uncertainties = np.asarray([record['sigma'] for record in records], dtype=float)
        if (not np.all(np.isfinite(observations)) or
                not np.all(np.isfinite(uncertainties)) or
                np.any(uncertainties <= 0)):
            raise ValueError('calibration values and uncertainties must be finite, with sigma > 0')

        def objective(viscosity):
            residual = (predict_at(experiments, viscosity) - observations) / uncertainties
            return float(np.dot(residual, residual))

        # The allowed interval is small.  A grid gives a reliable bracket,
        # including when a sparse calibration set places the optimum at an
        # endpoint; golden-section refinement then avoids an optimizer
        # dependency while retaining more precision than the instrument needs.
        lower, upper = 0.08, 0.18
        grid = np.linspace(lower, upper, 1001)
        scores = np.asarray([objective(value) for value in grid])
        best_index = int(np.argmin(scores))
        if best_index == 0 or best_index == len(grid) - 1:
            best = float(grid[best_index])
        else:
            left, right = float(grid[best_index - 1]), float(grid[best_index + 1])
            golden = (np.sqrt(5.0) - 1.0) / 2.0
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
            best = (left + right) / 2.0

        self.viscosity = float(best)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.viscosity)
