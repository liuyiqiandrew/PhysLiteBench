import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import solve_banded
from scipy.optimize import minimize_scalar

TAU2 = 0.4


def uniform_response(experiment, mobility):
    e = experiment
    width, ex, magnetic = e['width'], e['electric_field'], e['magnetic_field']
    viscosity = TAU2/(1+(2*TAU2*magnetic)**2)
    length = np.sqrt(mobility*viscosity)
    def integrated_velocity(y):
        return mobility*ex*(y-length*np.sinh(y/length)/np.cosh(width/(2*length)))
    if e['observable'] == 'current':
        return (integrated_velocity(width/2)-integrated_velocity(-width/2))/width
    a, b = np.asarray(e['contacts'])*width/2
    return float(magnetic*(1+2*TAU2/mobility)*(integrated_velocity(b)-integrated_velocity(a))
                 -2*TAU2*magnetic*ex*(b-a))


def response(experiment, mobility):
    e = experiment
    if e['field_gradient'] == 0 and e['field_modulation'] == 0:
        return uniform_response(e, mobility)
    width, ex = e['width'], e['electric_field']
    y = np.linspace(-width/2, width/2, 801)
    face = (y[:-1]+y[1:])/2
    dy = y[1]-y[0]
    def field(z):
        return (e['magnetic_field']+e['field_gradient']*2*z/width
                + e['field_modulation']*np.cos(2*np.pi*z/width))
    bf = field(face)
    viscosity = TAU2/(1+(2*TAU2*bf)**2)
    band = np.zeros((3, len(y)-2))
    band[1] = 1/mobility+(viscosity[:-1]+viscosity[1:])/dy**2
    band[0, 1:] = -viscosity[1:-1]/dy**2
    band[2, :-1] = -viscosity[1:-1]/dy**2
    velocity = np.r_[0., solve_banded((1,1), band, np.full(len(y)-2, ex)), 0.]
    if e['observable'] == 'current':
        return np.trapezoid(velocity, y)/width
    a, b = np.asarray(e['contacts'])*width/2
    primitive = cumulative_trapezoid(field(y)*velocity, y, initial=0)
    voltage = np.interp(b, y, primitive)-np.interp(a, y, primitive)
    normal_stress = 2*TAU2*bf*viscosity*np.diff(velocity)/dy
    # Quadratic continuation evaluates stress at wall contacts as well.
    from scipy.interpolate import CubicSpline
    stress = CubicSpline(face, normal_stress)
    voltage += stress(b)-stress(a)
    return float(voltage)


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        inputs = [r['input'] for r in records]
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def loss(mu):
            residual = (np.array([response(e, mu) for e in inputs])-values)/sigma
            return residual@residual
        self.mobility = float(minimize_scalar(loss, bounds=(0.05,1.5), method='bounded',
                                             options={'xatol': 1e-11}).x)
        return self

    def predict(self, experiments):
        return np.array([response(e, self.mobility) for e in experiments])
