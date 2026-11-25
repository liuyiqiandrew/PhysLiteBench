import numpy as np
from scipy.fft import dst, idst
from scipy.optimize import minimize_scalar
from scipy.integrate import solve_ivp
from scipy.sparse import diags

HEAT_CAPACITY = 1.5e6
RESISTIVITY = 2e-5
SEEBECK_300 = 2e-4
SEEBECK_SLOPE = 4e-6


def seebeck(temperature):
    return SEEBECK_300+SEEBECK_SLOPE*(temperature-300.)


def terminal_voltage(temperature, current, length):
    left, right = temperature[:, 0], temperature[:, -1]
    return (RESISTIVITY*current*length+SEEBECK_300*(right-left)
            +.5*SEEBECK_SLOPE*((right-300.)**2-(left-300.)**2))


def linear_solution(t, x, initial, current, boundary, conductivity):
    baseline = np.linspace(boundary[0], boundary[1], len(x))
    initial_modes = dst(initial[1:-1]-baseline[1:-1], type=1)
    spatial = 4*np.sin(np.arange(1, len(x)-1)*np.pi/(2*(len(x)-1)))**2/(x[1]-x[0])**2
    rates = conductivity*spatial/HEAT_CAPACITY
    forcing = dst(np.full(len(x)-2, RESISTIVITY*current**2/HEAT_CAPACITY), type=1)
    decay = np.exp(-t[:, None]*rates)
    modes = decay*initial_modes+(1-decay)*forcing/rates
    result = np.broadcast_to(baseline, (len(t), len(x))).copy()
    result[:, 1:-1] += idst(modes, type=1, axis=1)
    return result


def thermal_solution(t, x, initial, current, boundary, conductivity):
    if current == 0 or SEEBECK_SLOPE == 0:
        return linear_solution(t, x, initial, current, boundary, conductivity)
    if t[-1] == 0:
        return initial[None].copy()
    dx = x[1]-x[0]
    def derivative(time, interior):
        temperature = np.r_[boundary[0], interior, boundary[1]]
        curvature = (temperature[2:]-2*interior+temperature[:-2])/dx**2
        gradient = (temperature[2:]-temperature[:-2])/(2*dx)
        return (conductivity*curvature+RESISTIVITY*current**2
                -SEEBECK_SLOPE*current*interior*gradient)/HEAT_CAPACITY
    n = len(initial)-2
    pattern = diags([np.ones(n-abs(k)) for k in [-1, 0, 1]], [-1, 0, 1])
    answer = solve_ivp(derivative, (0., t[-1]), initial[1:-1], t_eval=t,
                       method='BDF', rtol=2e-10, atol=1e-8, jac_sparsity=pattern)
    if not answer.success:
        raise RuntimeError(answer.message)
    return np.column_stack([np.full(len(t), boundary[0]), answer.y.T,
                            np.full(len(t), boundary[1])])


class Model:
    def __init__(self):
        self.conductivity = None

    def fit(self, data):
        def loss(log_k):
            self.conductivity = float(np.exp(log_k))
            error = 0.
            for i, initial in enumerate(data['initial']):
                output = self.predict(data['t'], data['x'], initial, data['current'][i], data['boundary'][i])
                for field in ['temperature', 'voltage']:
                    error += np.sum(((output[field]-data[field][i])/data['sigma_'+field][i])**2)
            return error
        answer = minimize_scalar(loss, bounds=(np.log(.6), np.log(2.5)),
                                 method='bounded', options={'xatol': 1e-11})
        self.conductivity = float(np.exp(answer.x))
        return self

    def predict(self, t, x, initial, current, boundary):
        t, x, initial = np.asarray(t), np.asarray(x), np.asarray(initial)
        temperature = thermal_solution(t, x, initial, current, boundary, self.conductivity)
        return {'temperature': temperature,
                'voltage': terminal_voltage(temperature, current, x[-1]-x[0])}
