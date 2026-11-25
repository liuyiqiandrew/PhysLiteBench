from functools import lru_cache
import numpy as np

INPUT_FIELDS = ('force_x', 'force_y', 'drag_ratio', 'magnetic', 'temperature',
                'contrast', 'wavenumber', 'bath_contrast', 'switch_rate')


@lru_cache(None)
def spatial_grid(points):
    phase = np.arange(points)*2*np.pi/points
    modes = np.fft.fftfreq(points, 1/points)
    derivative = np.fft.ifft(1j*modes[:, None]*np.fft.fft(np.eye(points), axis=0), axis=0).real
    return phase, derivative


def positional_state(experiment, friction, points=129):
    phase, derivative = spatial_grid(points)
    derivative = derivative*experiment['wavenumber']
    temperature = experiment['temperature']*(1+experiment['contrast']*np.cos(phase))
    gamma_y = friction*experiment['drag_ratio']
    effective_force = experiment['force_x']+experiment['magnetic']*experiment['force_y']/gamma_y
    effective_drag = friction+experiment['magnetic']**2/gamma_y
    operator = -derivative@(effective_force*np.eye(points)-derivative*temperature[None, :])/effective_drag
    dx = 2*np.pi/points
    operator[-1, :] = dx
    rhs = np.zeros(points); rhs[-1] = 1
    density = np.linalg.solve(operator, rhs)
    current_x = (effective_force*density-derivative@(temperature*density))/effective_drag
    current_y = (experiment['force_y']*density-experiment['magnetic']*current_x)/gamma_y
    return phase, derivative, temperature, density, current_x, current_y, dx

def limiting_state(experiment, friction, points=129):
    phase, derivative, temperature, density, current_x, current_y, dx = positional_state(experiment, friction, points)
    temperatures = np.array([(1+sign*experiment['bath_contrast'])*temperature for sign in [1., -1.]])
    gamma_x, gamma_y = friction, friction*experiment['drag_ratio']
    magnetic, contact = experiment['magnetic'], experiment['switch_rate']
    force_x, force_y = experiment['force_x'], experiment['force_y']

    def moments(degree, lower, prior=None, higher=None):
        basis = [(p, degree-p) for p in range(degree, -1, -1)]
        lookup = {pair: i for i, pair in enumerate(basis)}
        relaxation = np.zeros((2*(degree+1), 2*(degree+1)))
        source = np.zeros((2, degree+1, points))
        for state in range(2):
            for i, (p, q) in enumerate(basis):
                row = state*(degree+1)+i
                relaxation[row, row] = -(p*gamma_x+q*gamma_y)-contact
                relaxation[row, (1-state)*(degree+1)+i] = contact
                if p:
                    relaxation[row, state*(degree+1)+lookup[p-1, q+1]] += p*magnetic
                if q:
                    relaxation[row, state*(degree+1)+lookup[p+1, q-1]] -= q*magnetic
                if p >= 2:
                    source[state, i] += p*(p-1)*gamma_x*temperatures[state]*lower[p-2, q][state]
                if q >= 2:
                    source[state, i] += q*(q-1)*gamma_y*temperatures[state]*lower[p, q-2][state]
                if prior is not None:
                    if p: source[state, i] += p*force_x*prior[p-1, q][state]
                    if q: source[state, i] += q*force_y*prior[p, q-1][state]
                if higher is not None:
                    source[state, i] -= derivative@higher[p+1, q][state]
        solved = np.linalg.solve(relaxation, -source.reshape(2*(degree+1), points)).reshape(2, degree+1, points)
        return {pair: solved[:, i] for i, pair in enumerate(basis)}

    zeroth = {(0, 0): np.array([density/2, density/2])}
    second = moments(2, zeroth)
    fourth = moments(4, second)
    first = moments(1, {}, zeroth, second)
    third = moments(3, first, second, fourth)
    density_correction = {(0, 0): np.zeros((2,points))}
    second_correction = moments(2, density_correction, first, third)
    entropy = dx*np.sum(gamma_x*(second_correction[2, 0]/temperatures-density_correction[0, 0])+
                        gamma_y*(second_correction[0, 2]/temperatures-density_correction[0, 0]))
    leak = dx*np.sum(gamma_x*(second[2, 0]/temperatures-zeroth[0, 0])+
                     gamma_y*(second[0, 2]/temperatures-zeroth[0, 0]))
    return dict(value=float(entropy), leak=float(leak), phase=phase, derivative=derivative,
                temperatures=temperatures, density=density, current_x=current_x, current_y=current_y,
                zeroth=zeroth, first=first, second=second, third=third, fourth=fourth,
                density_correction=density_correction, second_correction=second_correction, dx=dx)


@lru_cache(4096)
def entropy_rate(force_x, force_y, drag_ratio, magnetic, temperature, contrast,
                 wavenumber, bath_contrast, switch_rate, friction, detector_phase=None):
    # Equal contacts and zero magnetic field have exact inverse-drag scaling.
    if bath_contrast == 0. and magnetic == 0. and friction != 1.:
        return entropy_rate(force_x, force_y, drag_ratio, magnetic, temperature, contrast,
                            wavenumber, bath_contrast, switch_rate, 1., detector_phase)/friction
    e = dict(zip(INPUT_FIELDS, (force_x, force_y, drag_ratio, magnetic, temperature,
                                contrast, wavenumber, bath_contrast, switch_rate)))
    state=limiting_state(e,friction)
    if detector_phase is None:
        return state['value']
    weight=(1+np.cos(state['phase']-detector_phase))/2
    n0=state['density_correction'][0,0]
    n2=state['second_correction']
    temperatures=state['temperatures']
    local=np.sum(friction*(n2[2,0]/temperatures-n0)+friction*drag_ratio*(n2[0,2]/temperatures-n0),axis=0)
    return float(state['dx']*weight@local)


def coefficients(experiments, friction=1.):
    return np.array([entropy_rate(*(e[field] for field in INPUT_FIELDS), float(friction),e.get('detector_phase')) for e in experiments])


class Model:
    def __init__(self):
        self.friction = None

    def fit(self, records):
        """Fit the common friction coefficient to calibration measurements.

        The calibration contains independent Gaussian measurements, so the
        appropriate objective is the weighted residual sum of squares.  There
        is only one unknown parameter and its allowed interval is specified by
        the apparatus description; a bounded golden-section search keeps the
        implementation self-contained and avoids making the result depend on
        an optional optimisation package.
        """
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record['input'] for record in records]
        observed = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)
        if (observed.shape != sigma.shape or not np.isfinite(observed).all() or
                not np.isfinite(sigma).all() or np.any(sigma <= 0)):
            raise ValueError("calibration values and uncertainties must be finite, with sigma > 0")

        def objective(friction):
            predicted = coefficients(experiments, float(friction))
            residual = (predicted - observed)/sigma
            return float(residual @ residual)

        lower, upper = 0.7, 1.6
        # A small initial scan also makes the search robust if the objective
        # is nearly flat over part of the allowed interval.
        grid = np.linspace(lower, upper, 19)
        values = np.asarray([objective(point) for point in grid])
        best = int(np.argmin(values))
        if best == 0 or best == len(grid)-1:
            self.friction = float(grid[best])
            return self

        left, right = float(grid[best-1]), float(grid[best+1])
        golden = (np.sqrt(5.) - 1.)/2.
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

        candidate = (left + right)/2.
        # Include the scan points in the final comparison in case the
        # objective is not strictly unimodal for an unusual record set.
        candidates = [candidate, grid[best], x1, x2]
        self.friction = float(min(candidates, key=objective))
        return self

    def predict(self, experiments):
        if self.friction is None:
            raise ValueError("fit the model before requesting predictions")
        return coefficients(experiments, float(self.friction))
