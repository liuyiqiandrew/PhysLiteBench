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
def entropy_rate(force_x, force_y, drag_ratio, magnetic, temperature, contrast,
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
        experiments = [r['input'] for r in records]
        coefficient = coefficients(experiments)
        observed = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        inverse_friction = np.sum(coefficient*observed/sigma**2)/np.sum(coefficient**2/sigma**2)
        self.friction = float(np.clip(1/inverse_friction, .7, 1.6))
        return self

    def predict(self, experiments):
        return coefficients(experiments, self.friction)
