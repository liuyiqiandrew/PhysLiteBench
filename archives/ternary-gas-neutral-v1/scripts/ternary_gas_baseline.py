import numpy as np
from scipy.fft import dct, idct
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from scipy.sparse import diags

PAIR_RATIOS = np.array([[0., .2, 1.], [.2, 0., 5.], [1., 5., 0.]])


def binary_solution(t, x, initial, diffusivity):
    active = np.flatnonzero(np.max(initial, axis=0) > 1e-14)
    if len(active) != 2:
        return None
    count = len(x)-1
    rate = 4*np.sin(np.arange(count+1)*np.pi/(2*count))**2/(x[1]-x[0])**2
    rate *= diffusivity*PAIR_RATIOS[active[0], active[1]]
    modes = dct(initial, type=1, axis=0)
    return idct(np.exp(-t[:, None, None]*rate[None, :, None])*modes[None],
                type=1, axis=1)


class Model:
    def __init__(self):
        self.diffusivity = None

    def fit(self, data):
        def loss(log_d):
            self.diffusivity = float(np.exp(log_d))
            prediction = np.array([self.predict(data['t'], data['x'], initial)
                                   for initial in data['initial']])
            return np.sum(((prediction-data['mole_fraction'])/data['sigma'])**2)
        result = minimize_scalar(loss, bounds=(np.log(.005), np.log(.1)),
                                 method='bounded', options={'xatol': 1e-11})
        self.diffusivity = float(np.exp(result.x))
        return self

    def predict(self, t, x, initial):
        t, x, initial = np.asarray(t), np.asarray(x), np.asarray(initial)
        binary = binary_solution(t, x, initial, self.diffusivity)
        if binary is not None:
            return binary
        n = len(x)
        dx = x[1]-x[0]
        widths = np.full(n, dx)
        widths[[0, -1]] *= .5
        inverse = np.zeros((3, 3))
        mask = PAIR_RATIOS > 0
        inverse[mask] = 1/(self.diffusivity*PAIR_RATIOS[mask])

        def derivative(time, flat):
            fraction = flat.reshape(n, 3)
            face = (fraction[:-1]+fraction[1:])/2
            gradient = np.diff(fraction, axis=0)/dx
            coefficients = (1-face)/(face@inverse.T)
            current = -coefficients*gradient
            current -= face*np.sum(current, axis=1, keepdims=True)
            flux = np.zeros((n+1, 3))
            flux[1:-1] = current
            return (-np.diff(flux, axis=0)/widths[:, None]).ravel()

        if t[-1] == 0:
            return initial[None].copy()
        pattern = diags([np.ones(3*n-abs(k)) for k in range(-5, 6)],
                        range(-5, 6), format='csr')
        result = solve_ivp(derivative, (0., t[-1]), initial.ravel(), t_eval=t,
                           method='BDF', rtol=2e-8, atol=2e-10, jac_sparsity=pattern)
        if not result.success:
            raise RuntimeError(result.message)
        return result.y.T.reshape(len(t), n, 3)
