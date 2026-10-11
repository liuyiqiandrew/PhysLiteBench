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


def _critical_alpha(bath):
    """Return the first stability turning point on the regular branch.

    Let ``u = 4*pi*R**3*rho(R)/M`` and ``eta = G*M/(R*theta)``.
    The gas energy is ``M*theta*(u - 3/2)`` (the wall supplies the
    virial pressure), so for a bath of heat capacity ``b*M*k_B`` the
    conserved composite energy, apart from a constant factor, is

        (b + u - 3/2) / eta.

    Its first turning point is the finite-bath stability threshold.  In
    the imposed-temperature limit this reduces to the first maximum of
    eta, equivalently ``u == 1``.
    """
    if bath == 'infinite':
        return brentq(lambda x: milne(x)[0] - 1., 5., 12.)

    b = float(bath)

    def turning_condition(x):
        u, eta = milne(x)
        # u' = u*(3-u-eta)/x and eta' = eta*(u-1)/x.
        # This is the numerator of d[(b+u-3/2)/eta]/dx.
        return u*(3. - u - eta) - (b + u - 1.5)*(u - 1.)

    # The stated b range [0, 2] puts the first root between these
    # endpoints (the b=0 root is the usual isolated-gas threshold).
    return brentq(turning_condition, 5., 59.)


@lru_cache(maxsize=None)
def coefficient(observable, contrast, bath):
    if observable == 'temperature':
        alpha = brentq(lambda x: profile()(x)[0]-np.log(contrast), .01, 9.)
    else:
        alpha = _critical_alpha(bath)
    return 1/milne(alpha)[1]


def predict_at(experiments, gravity):
    return np.asarray([gravity*e['mass']/e['radius']*
                       coefficient(e['observable'], e.get('contrast'), e['bath_capacity'])
                       for e in experiments])


class Model:
    def __init__(self):
        self.gravity = None

    def fit(self, records):
        # Every response is G times a known dimensionless apparatus
        # factor.  Use the supplied Gaussian uncertainties for the
        # weighted one-parameter least-squares estimate.
        factors = []
        values = []
        weights = []
        for record in records:
            experiment = record['input']
            factors.append(
                experiment['mass'] / experiment['radius'] * coefficient(
                    experiment['observable'], experiment.get('contrast'),
                    experiment['bath_capacity']))
            values.append(record['value'])
            weights.append(1. / record['sigma']**2)

        factors = np.asarray(factors, dtype=float)
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        self.gravity = float(np.sum(weights * factors * values) /
                             np.sum(weights * factors * factors))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.gravity)
