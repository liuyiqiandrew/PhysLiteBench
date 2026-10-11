from functools import lru_cache
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


@lru_cache(maxsize=1)
def profile():
    start = 1e-5
    def derivative(x, y):
        return [y[1], np.exp(-y[0])-2*y[1]/x]
    return solve_ivp(derivative, (start, 60.), [start**2/6, start/3],
                     rtol=2e-11, atol=2e-13, dense_output=True, max_step=.1).sol


def milne(alpha):
    psi, gradient = profile()(alpha)
    eta = alpha*gradient
    return alpha**2*np.exp(-psi)/eta, eta


@lru_cache(maxsize=None)
def coefficient(observable, contrast, bath):
    if observable == 'temperature':
        alpha = brentq(lambda x: profile()(x)[0]-np.log(contrast), .01, 9.)
    elif bath == 'infinite':
        # At fixed temperature the canonical turning point is the maximum
        # of eta(alpha), or equivalently u(alpha) = 1.
        alpha = brentq(lambda x: milne(x)[0]-1, 5., 12.)
    else:
        # The gas energy, in units of G*M**2/R, is
        #
        #     e_g = (u - 3/2) / eta,
        #
        # where eta = G*M/(R*theta) and u is the usual Milne variable.
        # Adding a reservoir of heat capacity b*N*k_B adds b/eta.  For a
        # finite isolated gas-plus-reservoir composite, the first loss of
        # local stability is the first turning point of this total energy.
        b = float(bath)

        def energy_derivative(alpha):
            psi, gradient = profile()(alpha)
            density = np.exp(-psi)
            eta = alpha * gradient
            u = alpha * alpha * density / eta

            # eta' follows directly from the isothermal Lane-Emden ODE.
            eta_prime = alpha * density - gradient
            u_prime = u * (2.0 / alpha - gradient - eta_prime / eta)
            return (u_prime * eta - (b - 1.5 + u) * eta_prime) / eta**2

        # The derivative is negative on the dilute branch.  Find the first
        # sign change so later, oscillatory turning points are not selected.
        grid = np.linspace(.1, 59.0, 400)
        values = np.asarray([energy_derivative(x) for x in grid])
        changes = np.flatnonzero((values[:-1] <= 0.0) & (values[1:] > 0.0))
        if not len(changes):
            raise RuntimeError('Could not locate the finite-bath stability threshold')
        i = changes[0]
        alpha = brentq(energy_derivative, grid[i], grid[i + 1])
    return 1/milne(alpha)[1]


def predict_at(experiments, gravity):
    return np.asarray([gravity*e['mass']/e['radius']*
                       coefficient(e['observable'], e.get('contrast'), e['bath_capacity'])
                       for e in experiments])


class Model:
    def __init__(self):
        self.gravity = None

    def fit(self, records):
        design = []
        values = []
        weights = []
        for record in records:
            experiment = record['input']
            design.append(experiment['mass'] / experiment['radius'] *
                          coefficient(experiment['observable'],
                                      experiment.get('contrast'),
                                      experiment['bath_capacity']))
            values.append(record['value'])
            weights.append(1.0 / record['sigma']**2)

        design = np.asarray(design, dtype=float)
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        self.gravity = np.sum(weights * design * values) / np.sum(weights * design**2)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.gravity)
