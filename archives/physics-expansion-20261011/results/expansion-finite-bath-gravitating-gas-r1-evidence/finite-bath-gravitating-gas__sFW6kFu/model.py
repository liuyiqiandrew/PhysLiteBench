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
    else:
        alpha = critical_alpha(bath)
    return 1/milne(alpha)[1]


def _energy_derivative(alpha, bath):
    """Derivative of dimensionless composite energy along the branch.

    With eta=GM/(R*theta) and u=alpha**2 exp(-psi)/eta, the gas energy is
    M*theta*(u-3/2).  The bath contributes b*M*theta.  Apart from the
    constant GM/R, the composite energy is therefore

        (u - 3/2 + b) / eta.

    The first turning point of this quantity is the finite-bath stability
    threshold (the dilute branch is stable before this point).
    """
    psi, gradient = profile()(alpha)
    eta = alpha*gradient
    eta_prime = alpha*np.exp(-psi) - gradient
    u = alpha**2*np.exp(-psi)/eta
    u_prime = u*(2/alpha - gradient - eta_prime/eta)
    return u_prime/eta - (u - 1.5 + bath)*eta_prime/eta**2


@lru_cache(maxsize=None)
def critical_alpha(bath):
    """Return the first instability point for a specified reservoir."""
    if bath == 'infinite':
        # At imposed temperature the canonical threshold is the maximum of
        # eta(alpha), i.e. eta'(alpha)=0.
        return brentq(
            lambda x: x*np.exp(-profile()(x)[0]) - profile()(x)[1],
            5., 12.)

    bath = float(bath)
    if not 0 <= bath <= 2:
        raise ValueError('bath_capacity must be in [0, 2] or "infinite"')

    # The first zero lies between the canonical and microcanonical limits.
    return brentq(lambda x: _energy_derivative(x, bath), 8.99, 35.)


def predict_at(experiments, gravity):
    return np.asarray([gravity*e['mass']/e['radius']*
                       coefficient(e['observable'], e.get('contrast'), e['bath_capacity'])
                       for e in experiments])


class Model:
    def __init__(self):
        self.gravity = None

    def fit(self, records):
        # Every observation is linear in G.  The calibration supplies the
        # observation standard deviations, so use the corresponding weighted
        # least-squares estimate rather than relying on any one record.
        factors = []
        values = []
        weights = []
        for record in records:
            experiment = record['input']
            factors.append(
                experiment['mass'] / experiment['radius'] *
                coefficient(experiment['observable'],
                            experiment.get('contrast'),
                            experiment['bath_capacity']))
            values.append(record['value'])
            weights.append(1.0 / record['sigma']**2)

        factors = np.asarray(factors, dtype=float)
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        denominator = np.sum(weights * factors**2)
        if factors.size == 0 or not np.isfinite(denominator) or denominator <= 0:
            raise ValueError('records must contain at least one valid calibration')
        self.gravity = float(np.sum(weights * factors * values) / denominator)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.gravity)
