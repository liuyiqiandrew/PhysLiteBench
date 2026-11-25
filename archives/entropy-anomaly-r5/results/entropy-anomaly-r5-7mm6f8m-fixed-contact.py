from functools import lru_cache
import numpy as np


@lru_cache(None)
def spatial_grid(points):
    x = np.arange(points)*2*np.pi/points
    modes = np.fft.fftfreq(points, 1/points)
    derivative = np.fft.ifft(1j*modes[:, None]*np.fft.fft(np.eye(points), axis=0), axis=0).real
    return x, derivative


def positional_state(experiment, friction, points=129):
    x, derivative = spatial_grid(points)
    derivative = derivative*experiment['wavenumber']
    base = experiment['temperature']
    contrast = experiment['contrast']
    bath_contrast = experiment['bath_contrast']
    temperatures = np.array([base*(1+sign*bath_contrast)*(1+contrast*np.cos(x)) for sign in [1, -1]])
    gradients = np.array([-base*(1+sign*bath_contrast)*contrast*experiment['wavenumber']*np.sin(x) for sign in [1, -1]])
    gamma_y = friction*experiment['drag_ratio']
    magnetic = experiment['magnetic']
    effective_force = experiment['force_x']+magnetic*experiment['force_y']/gamma_y
    effective_drag = friction+magnetic**2/gamma_y
    switch_rate = experiment['switch_rate']
    operators = [-(derivative@(effective_force*np.eye(points)-derivative*temp[None, :]))/effective_drag for temp in temperatures]
    coupling = switch_rate*np.eye(points)
    system = np.block([[operators[0]-coupling, coupling], [coupling, operators[1]-coupling]])
    system[-1, :] = 2*np.pi/points
    rhs = np.zeros(2*points)
    rhs[-1] = 1
    density = np.linalg.solve(system, rhs).reshape(2, points)
    current_x = np.array([(effective_force*rho-derivative@(temp*rho))/effective_drag for rho, temp in zip(density, temperatures)])
    current_y = (experiment['force_y']*density-magnetic*current_x)/gamma_y
    return temperatures, gradients, density, current_x, current_y, 2*np.pi/points


def heat_coefficient(friction, drag_ratio, magnetic):
    gamma_y = friction*drag_ratio
    relaxation = np.diag([3*friction, 2*friction+gamma_y, friction+2*gamma_y, 3*gamma_y])
    for i in range(3):
        relaxation[i, i+1] = -magnetic*np.sqrt((3-i)*(i+1))
        relaxation[i+1, i] = -relaxation[i, i+1]
    source = np.array([np.sqrt(6), 0., np.sqrt(2), 0.])
    return float(source@np.linalg.solve(relaxation, source)/4)


INPUT_FIELDS = ('force_x', 'force_y', 'drag_ratio', 'magnetic', 'temperature', 'contrast', 'wavenumber', 'bath_contrast', 'switch_rate')


@lru_cache(None)
def entropy_rate(force_x, force_y, drag_ratio, magnetic, temperature, contrast, wavenumber, bath_contrast, switch_rate, friction):
    e = dict(zip(INPUT_FIELDS, (force_x, force_y, drag_ratio, magnetic, temperature, contrast, wavenumber, bath_contrast, switch_rate)))
    temp, gradient, density, current_x, current_y, dx = positional_state(e, friction)
    gamma_y = friction*drag_ratio
    positional = np.sum((friction*current_x**2+gamma_y*current_y**2)/(temp*density))*dx
    positional += switch_rate*np.sum(density*np.log(density/density[::-1]))*dx
    thermal = heat_coefficient(friction, drag_ratio, magnetic)*np.sum(density*gradient**2/temp)*dx
    return float(positional+thermal+2*switch_rate*bath_contrast**2/(1-bath_contrast**2))


def coefficients(experiments, friction=1.):
    return np.array([entropy_rate(*(e[field] for field in INPUT_FIELDS), friction) for e in experiments])


class Model:
    def __init__(self):
        self.friction = None

    def fit(self, records):
        """Fit the common drag coefficient from the calibration measurements.

        The calibration errors are independent Gaussian errors, so the
        appropriate objective is the uncertainty-weighted sum of squared
        residuals.  Friction is the only fitted quantity and the documented
        calibration range is used as a hard bound.  A small, dependency-free
        golden-section search is sufficient here because this is a bounded
        one-dimensional problem.
        """
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record['input'] for record in records]
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)
        if (values.ndim != 1 or sigma.ndim != 1 or
                values.shape != sigma.shape or
                not np.isfinite(values).all() or
                not np.isfinite(sigma).all() or
                np.any(sigma <= 0)):
            raise ValueError("calibration values and sigma must be finite, positive arrays")

        def objective(friction):
            prediction = coefficients(experiments, float(friction))
            residual = (prediction - values) / sigma
            return float(residual @ residual)

        lower, upper = 0.7, 1.6
        golden = (np.sqrt(5.0) - 1.0) / 2.0
        left = lower
        right = upper
        x1 = right - golden * (right - left)
        x2 = left + golden * (right - left)
        y1 = objective(x1)
        y2 = objective(x2)

        # The forward model is smooth in friction.  This also avoids relying
        # on scipy being present in the evaluation environment.
        for _ in range(64):
            if y1 <= y2:
                right, x2, y2 = x2, x1, y1
                x1 = right - golden * (right - left)
                y1 = objective(x1)
            else:
                left, x1, y1 = x1, x2, y2
                x2 = left + golden * (right - left)
                y2 = objective(x2)

        candidates = (lower, upper, x1, x2, (left + right) / 2.0)
        self.friction = float(min(candidates, key=objective))
        return self

    def predict(self, experiments):
        if self.friction is None:
            raise RuntimeError("fit must be called before predict")
        prediction = coefficients(experiments, float(self.friction))
        return np.asarray(prediction, dtype=float).reshape(-1)
