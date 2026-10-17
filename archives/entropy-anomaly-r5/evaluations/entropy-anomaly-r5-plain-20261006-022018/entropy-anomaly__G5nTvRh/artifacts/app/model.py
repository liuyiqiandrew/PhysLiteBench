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
    temperatures = np.array([base*(1+sign*bath_contrast)*(1+contrast*np.cos(wavenumber*x)) for sign in [1, -1]])
    gradients = np.array([-base*(1+sign*bath_contrast)*contrast*wavenumber*np.sin(wavenumber*x) for sign in [1, -1]])
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
    temp, _, density, current_x, current_y, dx = positional_state(e, friction)

    # Write G for the deterministic velocity relaxation as
    #
    #     G = [[gamma_x, -magnetic], [magnetic, gamma_y]].
    #
    # If M_s is the (unnormalised) second velocity moment in contact state
    # s, its small-m expansion is
    #
    #     M_s = T_s*rho_s*I/m + N_s + O(m).
    #
    # The bath entropy rate is the temperature-weighted trace of N_s.  The
    # equations below are the moment hierarchy at the first non-vanishing
    # order.  Keeping this form is important: in particular, switching
    # between two different temperatures transfers kinetic heat even when
    # the positional state is uniform.
    points = density.shape[1]
    _, derivative = spatial_grid(points)
    gamma_x = friction
    gamma_y = friction*drag_ratio
    force_x = e['force_x']
    force_y = e['force_y']
    magnetic = e['magnetic']
    switch_rate = e['switch_rate']
    currents = np.stack((current_x, current_y), axis=-1)

    # R is m times the leading third velocity moment.  Its four independent
    # symmetric components are (xxx, xxy, xyy, yyy).
    d_temperature_moment = derivative@(temp**2*density).T
    d_temperature_moment = d_temperature_moment.T
    rhs_r = np.empty((2, points, 4))
    rhs_r[:, :, 0] = (
        -3*d_temperature_moment
        + 3*temp*density*force_x
        + 6*temp*gamma_x*currents[:, :, 0]
    )
    rhs_r[:, :, 1] = (
        temp*density*force_y
        + 2*temp*gamma_x*currents[:, :, 1]
    )
    rhs_r[:, :, 2] = (
        -d_temperature_moment
        + temp*density*force_x
        + 2*temp*gamma_y*currents[:, :, 0]
    )
    rhs_r[:, :, 3] = (
        3*temp*density*force_y
        + 6*temp*gamma_y*currents[:, :, 1]
    )
    relaxation_r = np.array([
        [3*gamma_x, -3*magnetic, 0., 0.],
        [magnetic, 2*gamma_x + gamma_y, -2*magnetic, 0.],
        [0., 2*magnetic, gamma_x + 2*gamma_y, -magnetic],
        [0., 0., 3*magnetic, 3*gamma_y],
    ])
    third_moment = np.linalg.solve(
        relaxation_r, rhs_r.reshape(-1, 4).T
    ).T.reshape(2, points, 4)

    d_third_x = np.empty((2, points, 3))
    d_third_x[:, :, 0] = (derivative@third_moment[:, :, 0].T).T
    d_third_x[:, :, 1] = (derivative@third_moment[:, :, 1].T).T
    d_third_x[:, :, 2] = (derivative@third_moment[:, :, 2].T).T
    exchanged_temperature = (
        switch_rate*(temp[::-1]*density[::-1] - temp*density)
    )

    # Q = G*N + N*G.T, represented by (xx, xy, yy).
    rhs_n = np.empty((2, points, 3))
    rhs_n[:, :, 0] = (
        -d_third_x[:, :, 0]
        + 2*force_x*currents[:, :, 0]
        + exchanged_temperature
    )
    rhs_n[:, :, 1] = (
        -d_third_x[:, :, 1]
        + force_x*currents[:, :, 1]
        + force_y*currents[:, :, 0]
    )
    rhs_n[:, :, 2] = (
        -d_third_x[:, :, 2]
        + 2*force_y*currents[:, :, 1]
        + exchanged_temperature
    )
    relaxation_n = np.array([
        [2*gamma_x, -2*magnetic, 0.],
        [magnetic, gamma_x + gamma_y, -magnetic],
        [0., 2*magnetic, 2*gamma_y],
    ])
    second_correction = np.linalg.solve(
        relaxation_n, rhs_n.reshape(-1, 3).T
    ).T.reshape(2, points, 3)
    entropy = np.sum(
        (gamma_x*second_correction[:, :, 0]
         + gamma_y*second_correction[:, :, 2]) / temp
    )*dx
    return float(entropy)


def coefficients(experiments, friction=1.):
    return np.array([entropy_rate(*(e[field] for field in INPUT_FIELDS), friction) for e in experiments])


class Model:
    def __init__(self):
        self.friction = None

    def fit(self, records):
        if not records:
            raise ValueError("records must contain at least one calibration record")

        experiments = [record['input'] for record in records]
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)
        if (values.shape != sigmas.shape or not np.isfinite(values).all()
                or not np.isfinite(sigmas).all() or np.any(sigmas <= 0)):
            raise ValueError("calibration values and uncertainties must be finite, positive arrays")

        def objective(friction):
            prediction = coefficients(experiments, float(friction))
            residual = (prediction - values)/sigmas
            return float(residual@residual)

        # The calibration is one-dimensional and the physical interval is
        # given by the apparatus.  A short scan makes the fit robust to the
        # mild numerical curvature of the discretised stationary solve;
        # golden-section refinement then avoids any external dependency.
        lower, upper = 0.7, 1.6
        scan = np.linspace(lower, upper, 17)
        scan_values = np.array([objective(friction) for friction in scan])
        best = int(np.argmin(scan_values))
        left = scan[max(0, best - 1)]
        right = scan[min(scan.size - 1, best + 1)]
        if left == right:
            self.friction = float(scan[best])
            return self

        golden = (np.sqrt(5.) - 1.)/2.
        x1 = right - golden*(right-left)
        x2 = left + golden*(right-left)
        f1, f2 = objective(x1), objective(x2)
        for _ in range(32):
            if f1 > f2:
                left, x1, f1 = x1, x2, f2
                x2 = left + golden*(right-left)
                f2 = objective(x2)
            else:
                right, x2, f2 = x2, x1, f1
                x1 = right - golden*(right-left)
                f1 = objective(x1)
        self.friction = float((left + right)/2)
        return self

    def predict(self, experiments):
        if self.friction is None:
            raise ValueError("fit must be called before predict")
        return coefficients(experiments, self.friction)
