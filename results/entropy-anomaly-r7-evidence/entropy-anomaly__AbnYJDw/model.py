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
        weight = None
    else:
        weight=(1+np.cos(state['phase']-detector_phase))/2

    # The finite part of the background preparation is normally zero (the
    # uniform, force-free particle has no spatial or mechanical driving), but
    # subtract it explicitly.  This is also useful numerically: it keeps the
    # implementation tied to the experiment definition rather than relying on
    # that special-case cancellation.
    background = dict(e)
    background.update(force_x=0., force_y=0., contrast=0.)
    background_value = limiting_state(background, friction)['value']

    if weight is None:
        residence = 1.
        result = state['value']
    else:
        residence = float(state['dx']*weight@state['density'])
        n0=state['density_correction'][0,0]
        n2=state['second_correction']
        temperatures=state['temperatures']
        local=np.sum(friction*(n2[2,0]/temperatures-n0)+
                     friction*drag_ratio*(n2[0,2]/temperatures-n0),axis=0)
        result = float(state['dx']*weight@local)

    return float(result-background_value*residence)


def coefficients(experiments, friction=1.):
    return np.array([entropy_rate(*(e[field] for field in INPUT_FIELDS), float(friction),e.get('detector_phase')) for e in experiments])


class Model:
    def __init__(self):
        self.friction = None

    def fit(self, records):
        """Fit the one shared friction coefficient from calibration records.

        In the documented calibration set, bath_contrast and magnetic are
        both zero.  In that case the limiting entropy rate has the exact
        scaling ``rate(friction) = rate(1) / friction``.  The weighted
        least-squares fit can therefore be done directly in the variable
        ``1 / friction`` instead of repeatedly solving the moment system.
        A small bounded one-dimensional fallback keeps the method useful for
        otherwise well-formed records too.
        """
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        inputs = [record['input'] for record in records]
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)
        if (not np.isfinite(values).all() or not np.isfinite(sigmas).all() or
                np.any(sigmas <= 0)):
            raise ValueError("calibration values and uncertainties must be finite")

        documented = all(float(item['magnetic']) == 0. and
                         float(item['bath_contrast']) == 0.
                         for item in inputs)
        if documented:
            unit_rates = coefficients(inputs, 1.)
            weights = 1. / (sigmas*sigmas)
            denominator = float(np.sum(weights*unit_rates*unit_rates))
            if denominator > 0. and np.isfinite(denominator):
                inverse_friction = float(np.sum(weights*unit_rates*values) / denominator)
                if np.isfinite(inverse_friction) and inverse_friction > 0.:
                    self.friction = float(np.clip(1./inverse_friction, .7, 1.6))
                    return self

        # Generic bounded weighted least squares for nonstandard records.
        def objective(friction):
            prediction = coefficients(inputs, friction)
            return float(np.sum(((prediction-values)/sigmas)**2))

        lo, hi = .7, 1.6
        # Golden-section search is sufficient because this is a single shared
        # scalar and avoids adding a dependency just for fitting it.
        golden = (np.sqrt(5.)-1.)/2.
        a, b = lo, hi
        c, d = b-golden*(b-a), a+golden*(b-a)
        fc, fd = objective(c), objective(d)
        for _ in range(60):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b-golden*(b-a)
                fc = objective(c)
            else:
                a, c, fc = c, d, fd
                d = a+golden*(b-a)
                fd = objective(d)
        self.friction = float((a+b)/2.)
        return self

    def predict(self, experiments):
        return coefficients(experiments, self.friction)
