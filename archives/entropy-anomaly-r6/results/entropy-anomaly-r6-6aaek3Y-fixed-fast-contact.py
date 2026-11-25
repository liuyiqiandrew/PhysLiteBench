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

def heat_coefficient(friction, drag_ratio, magnetic):
    gamma_y = friction*drag_ratio
    relaxation = np.diag([3*friction, 2*friction+gamma_y, friction+2*gamma_y, 3*gamma_y])
    for i in range(3):
        relaxation[i, i+1] = -magnetic*np.sqrt((3-i)*(i+1))
        relaxation[i+1, i] = -relaxation[i, i+1]
    source = np.array([np.sqrt(6), 0., np.sqrt(2), 0.])
    return float(source@np.linalg.solve(relaxation, source)/4)


@lru_cache(4096)
def mixed_contact_entropy_rate(force_x, force_y, drag_ratio, magnetic, temperature, contrast,
                 wavenumber, bath_contrast, switch_rate, friction):
    if bath_contrast == 0. and magnetic == 0. and friction != 1.:
        return entropy_rate(force_x, force_y, drag_ratio, magnetic, temperature, contrast,
                            wavenumber, bath_contrast, switch_rate, 1.)/friction
    e = dict(zip(INPUT_FIELDS, (force_x, force_y, drag_ratio, magnetic, temperature,
                                contrast, wavenumber, bath_contrast, switch_rate)))
    phase, _, temp, density, current_x, current_y, dx = positional_state(e, friction)
    gradient = -temperature*contrast*wavenumber*np.sin(phase)
    entropy = np.sum((friction*current_x**2+friction*drag_ratio*current_y**2)/(temp*density))*dx
    entropy += heat_coefficient(friction, drag_ratio, magnetic)*np.sum(density*gradient**2/temp)*dx
    return float(entropy/(1-bath_contrast**2))


def coefficients(experiments, friction=1.):
    return np.array([entropy_rate(*(e[field] for field in INPUT_FIELDS), float(friction)) for e in experiments])


class Model:
    def __init__(self):
        self.friction = None

    def fit(self, records):
        """Fit the one apparatus parameter from the calibration measurements.

        The calibration values are independent measurements with supplied
        standard errors, so the natural fit is a (weighted) one-dimensional
        least-squares fit.  Keeping the search one-dimensional also avoids
        making the result depend on an optional optimisation package.
        """
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        inputs = []
        observed = np.empty(len(records), dtype=float)
        sigma = np.empty(len(records), dtype=float)
        for i, record in enumerate(records):
            try:
                inputs.append(record['input'])
                observed[i] = float(record['value'])
                sigma[i] = float(record['sigma'])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid calibration record") from exc
        if (not np.isfinite(observed).all() or
                not np.isfinite(sigma).all() or (sigma <= 0).any()):
            raise ValueError("calibration values and uncertainties must be finite")

        def objective(friction):
            prediction = coefficients(inputs, float(friction))
            residual = (prediction - observed) / sigma
            return float(residual @ residual)

        # The documented range is closed.  A bounded golden-section search is
        # sufficient here because this is a smooth scalar nuisance parameter;
        # evaluating the endpoints also handles a fit whose optimum is on a
        # boundary.
        lo, hi = 0.7, 1.6
        golden = (np.sqrt(5.0) - 1.0) / 2.0
        x1 = hi - golden * (hi - lo)
        x2 = lo + golden * (hi - lo)
        f1, f2 = objective(x1), objective(x2)
        for _ in range(80):
            if f1 <= f2:
                hi, x2, f2 = x2, x1, f1
                x1 = hi - golden * (hi - lo)
                f1 = objective(x1)
            else:
                lo, x1, f1 = x1, x2, f2
                x2 = lo + golden * (hi - lo)
                f2 = objective(x2)
        candidates = (0.7, 1.6, (lo + hi) / 2.0)
        self.friction = float(min(candidates, key=objective))
        return self

    def predict(self, experiments):
        if self.friction is None:
            raise ValueError("fit the model before calling predict")
        return coefficients(experiments, self.friction)


def finite_contact_state(experiment, friction, points=129):
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
    antisymmetric_density = -derivative@first[1, 0][0]/(2*contact)
    density_correction = {(0, 0): np.array([antisymmetric_density, -antisymmetric_density])}
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
def entropy_rate(force_x,force_y,drag_ratio,magnetic,temperature,contrast,wavenumber,bath_contrast,switch_rate,friction):
    if bath_contrast==0. or contrast==0.:
        return mixed_contact_entropy_rate(force_x,force_y,drag_ratio,magnetic,temperature,contrast,wavenumber,bath_contrast,switch_rate,friction)
    experiment=dict(zip(INPUT_FIELDS,(force_x,force_y,drag_ratio,magnetic,temperature,contrast,wavenumber,bath_contrast,switch_rate)))
    return finite_contact_state(experiment,friction)['value']
