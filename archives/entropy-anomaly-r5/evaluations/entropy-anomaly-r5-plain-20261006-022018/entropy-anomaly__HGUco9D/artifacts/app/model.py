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
    wavenumber = experiment['wavenumber']
    base = experiment['temperature']
    contrast = experiment['contrast']
    bath_contrast = experiment['bath_contrast']
    temperatures = np.array([
        base*(1+sign*bath_contrast)*(1+contrast*np.cos(wavenumber*x))
        for sign in [1, -1]
    ])
    gradients = np.array([
        -base*(1+sign*bath_contrast)*contrast*wavenumber*np.sin(wavenumber*x)
        for sign in [1, -1]
    ])
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
    thermal = heat_coefficient(friction, drag_ratio, magnetic)*np.sum(density*gradient**2/temp)*dx

    # A switch does no instantaneous mechanical work, but velocity remains
    # continuous.  During the O(m) velocity re-equilibration after a switch,
    # the new bath absorbs/releases the old kinetic energy.  In two dimensions
    # its limiting entropy contribution is
    #
    #   k p_old (T_old-T_new)/T_new.
    #
    # This is heat recorded by the reservoirs, not the entropy production of
    # an abstract Markov label (which would incorrectly introduce a
    # p*log(p/p_other) term here).
    old_temperature = temp[0]
    new_temperature = temp[1]
    switch_heat = switch_rate*np.sum(
        density[0]*(old_temperature-new_temperature)/new_temperature
        + density[1]*(new_temperature-old_temperature)/old_temperature
    )*dx
    return float(positional+thermal+switch_heat)


def coefficients(experiments, friction=1.):
    return np.array([entropy_rate(*(e[field] for field in INPUT_FIELDS), friction) for e in experiments])


class Model:
    def __init__(self):
        self.friction = None

    def fit(self, records):
        """Estimate the shared friction by weighted least squares.

        The calibration values have independent Gaussian uncertainties, so
        the likelihood is a one-dimensional weighted least-squares problem.
        A bounded golden-section search avoids adding a fitting dependency and
        keeps the estimate inside the documented physical range.
        """
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record['input'] for record in records]
        observed = np.asarray([record['value'] for record in records], dtype=float)
        uncertainty = np.asarray([record['sigma'] for record in records], dtype=float)
        if (not np.isfinite(observed).all() or
                not np.isfinite(uncertainty).all() or
                np.any(uncertainty <= 0)):
            raise ValueError("calibration values and uncertainties must be finite, with sigma > 0")

        def objective(friction):
            residual = (coefficients(experiments, float(friction))-observed)/uncertainty
            return float(residual@residual)

        # The apparatus specifies this interval for the unknown coefficient.
        left, right = 0.7, 1.6
        golden = (np.sqrt(5.0)-1.0)/2.0
        x_left = right-golden*(right-left)
        x_right = left+golden*(right-left)
        f_left = objective(x_left)
        f_right = objective(x_right)
        for _ in range(48):
            if f_left <= f_right:
                right, x_right, f_right = x_right, x_left, f_left
                x_left = right-golden*(right-left)
                f_left = objective(x_left)
            else:
                left, x_left, f_left = x_left, x_right, f_right
                x_right = left+golden*(right-left)
                f_right = objective(x_right)

        self.friction = float((left+right)/2.0)
        return self

    def predict(self, experiments):
        friction = 1.0 if self.friction is None else self.friction
        return np.asarray(coefficients(experiments, friction), dtype=float)
