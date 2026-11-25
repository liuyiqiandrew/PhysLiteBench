"""Heat-fluctuation large-deviation rate for a circulating trap.

The particle obeys the two-dimensional overdamped SDE

    dx = mu * (-k*x + k*a*R*x) dt + sqrt(2*mu*T) dW,   R = [[0,1],[-1,0]],

with T = 1. Up to a positive Stratonovich/Lebesgue correction, the heat into
the bath is Q(t) = int f(x) . circ(dx) with f(x) = -k*x + k*a*R*x.

The reported inverse-time rate I(z) with z = Q / (q_bar t) is the Fenchel-
Legendre transform of the Feynman-Kac cumulant rate Lambda(theta) of Q(t):

    Lambda(theta) = mu*k*(1 - sqrt(1 - 4*a^2*theta*(1+theta))),
    q_bar         = Lambda'(0) = 2*mu*k*a^2,
    I(k,a,z)      = sup_theta [ q_bar*z*theta - Lambda(theta) ],  theta in
                    [theta_-, theta_+] where 1 - 4 a^2 th(1+th) >= 0.

The supremum has the closed form

    I = mu*k*( sqrt((1+a^2)*(1+a^2*z^2)) - a^2*z - 1 ).

It vanishes at z = 1 (the mean of Z_t is one), is convex in z, and is valid
on the whole reported domain.  Only mobility mu is unknown; a, k are inputs.
"""

import numpy as np


def response(e):
    """Unit-mobility rate I/(mu) for one input dict (stiffness, drive, z)."""
    k, a, z = e['stiffness'], e['drive_ratio'], e['relative_heat']
    return k * (np.sqrt((1.0 + a * a) * (1.0 + a * a * z * z))
                - a * a * z - 1.0)


def predict_at(experiments, mobility):
    """Rate predictions for a list of inputs, scaled by the common mobility."""
    base = np.array([response(e) for e in experiments], dtype=float)
    return mobility * base


class Model:
    """Fit the common mobility from calibration, then predict I(z)."""

    def __init__(self):
        self.mobility = None

    def fit(self, records):
        """Weighted least squares (== Gaussian MLE, known sigmas) for mu."""
        base = np.array([response(r['input']) for r in records], dtype=float)
        value = np.array([r['value'] for r in records], dtype=float)
        sigma = np.array([r['sigma'] for r in records], dtype=float)
        weights = 1.0 / (sigma * sigma)
        denom = np.sum(weights * base * base)
        if denom > 0.0:
            self.mobility = float(np.sum(weights * base * value) / denom)
        else:
            self.mobility = 1.0
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
