"""Heat-fluctuation rate for a particle in a circulating harmonic trap.

Apparatus (see README.md): overdamped 2D dynamics

    dx = mobility * [-stiffness*x + stiffness*drive_ratio*R*x] dt
         + sqrt(2*mobility*T) dW,      R = [[0, 1], [-1, 0]],

with T = 1.  The reported quantity is the inverse-time large-deviation rate
of the (q_bar-normalised) heat Q(t),

    I(z) = -lim_{eps->0} lim_{t->inf} log P(|Z_t - z| < eps) / t,
    Z_t  = Q(t) / (q_bar * t).

Modelling the heat as a quadratic functional of the linear (Gaussian) process,
the scaled cumulant generating function of Q is obtained from the top
eigenvalue of the tilted generator,

    lambda(s) = mobility*stiffness*[1 - sqrt(1 - 4 r^2 s (1 + s))],   r = drive_ratio.

Its Legendre transform gives the rate function

    I(z) = mobility * stiffness * [ sqrt((1 + r^2)(1 + r^2 z^2)) - r^2 z - 1 ],

which is zero at z = 1 (the normalising stationary mean rate) and obeys the
Gallavotti-Cohen symmetry I(z) - I(-z) = -q_bar z.  Only the common, constant
mobility is unknown; it enters I as an overall factor because the heat is
linear in the drift while q_bar is proportional to the mobility.
"""

import numpy as np


def response(e):
    """Mobility-free rate I(z)/mobility for one preparation."""
    stiffness = e['stiffness']
    r = e['drive_ratio']
    z = e['relative_heat']
    r2 = r * r
    rate = np.sqrt((1.0 + r2) * (1.0 + r2 * z * z)) - r2 * z - 1.0
    return stiffness * rate


def predict_at(experiments, mobility):
    base = np.asarray([response(e) for e in experiments], dtype=float)
    return mobility * base


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        """Infer the common mobility from calibration readings.

        Each record carries a measured `value` of I with known standard
        deviation `sigma`; the response is linear in the mobility, so the
        weighted least-squares (Gaussian maximum-likelihood) estimate is
        closed form.
        """
        records = list(records)
        if not records:
            raise ValueError('fit() requires at least one calibration record.')

        base = np.asarray([response(r['input']) for r in records], dtype=float)
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r.get('sigma', 1.0) for r in records], dtype=float)

        weights = 1.0 / np.square(sigma)
        denom = np.sum(weights * base * base)
        self.mobility = float(np.sum(weights * base * values) / denom)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
