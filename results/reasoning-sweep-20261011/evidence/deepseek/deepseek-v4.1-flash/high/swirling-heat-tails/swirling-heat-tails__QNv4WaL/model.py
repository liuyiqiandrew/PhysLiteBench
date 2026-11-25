"""Heat-fluctuation large-deviation rate for a circulating overdamped trap.

Apparatus (see README):
    dx = mobility * [-stiffness*x + stiffness*drive_ratio*R*x] dt
         + sqrt(2*mobility*T) dW,      R = [[0, 1], [-1, 0]],
    Q(t) = integral_0^t [-stiffness*x + stiffness*drive_ratio*R*x] . circ(dx),
    Z_t  = Q(t) / (q_bar * t),   q_bar = lim E[Q(t)]/t > 0,
    I(z) = -lim_{eps->0} lim_{t->inf} log P(|Z_t - z| < eps) / t.

Derivation of the reported rate
-------------------------------
Let k = stiffness, a = drive_ratio, mu = mobility.  With A = k(-I + a R) the
drift is mu*A*x and the integrand of Q is A*x.  Because the noise is isotropic
and the problem is invariant under time reparametrisation tau = mu*t, Z_t is
independent of mu and the rate factorises as I(z) = mu * f(k, a, z).

Splitting A into its symmetric (-k I) and antisymmetric (k a R) parts, the
Stratonovich integral of the symmetric part is an exact differential and the
antisymmetric part carries no Ito correction, so

    Q(t) = -(k/2)(|x_t|^2 - |x_0|^2) + k a integral_0^t (R x) . dx .

The endpoint terms are O(1) and do not affect the t -> inf rate.  The Ito
functional above tilts the Ornstein-Uhlenbeck generator L = mu*Delta + (Bx).grad,
B = mu k (-I + a R), into

    L_nu = L + mu k^2 a^2 nu (nu - 1) |x|^2 .

Looking for the top eigenfunction e^{-x^T P x / 2} with P = p I (the
rotation-symmetric branch) gives the algebraic Riccati equation

    mu p^2 + c_nu = -mu k p,     c_nu = mu k^2 a^2 nu (nu - 1),

whose solution p = (k/2)[-1 + sqrt(1 - 4 a^2 nu(nu-1))] yields the scaled
cumulant generating function of Q(t) (with theta = -nu)

    Lambda(theta) = mu k [1 - sqrt(1 - 4 a^2 theta (theta + 1))],
    q_bar         = -Lambda'(0) = 2 mu k a^2.

Legendre transforming at the optimal theta, z = (2 theta + 1)/sqrt(1-4a^2 theta(theta+1)),
gives a strictly convex rate with minimum 0 at z = 1:

    I(z) = mu * k * [ sqrt((1 + a^2)(1 + a^2 z^2)) - a^2 z - 1 ].

This holds for every z in the stated domain: the optimal theta always lies in
the real (non-imaginary-sqrt) window of Lambda.
"""

import numpy as np


def _unit_rate(stiffness, drive_ratio, relative_heat):
    """Rate per unit mobility: f(k, a, z) with I(z) = mobility * f(k, a, z)."""
    a = np.asarray(drive_ratio, dtype=float)
    z = np.asarray(relative_heat, dtype=float)
    root = np.sqrt((1.0 + a * a) * (1.0 + a * a * z * z))
    return np.asarray(stiffness, dtype=float) * (root - a * a * z - 1.0)


def response(e):
    """Unit-mobility rate for one input dictionary (no mobility factor)."""
    return float(_unit_rate(e['stiffness'], e['drive_ratio'], e['relative_heat']))


def predict_at(experiments, mobility):
    if len(experiments) == 0:
        return np.empty(0)
    k = np.array([e['stiffness'] for e in experiments], dtype=float)
    a = np.array([e['drive_ratio'] for e in experiments], dtype=float)
    z = np.array([e['relative_heat'] for e in experiments], dtype=float)
    return float(mobility) * _unit_rate(k, a, z)


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        """Infer the common mobility from calibration records.

        Each record has ``input`` (a dict), a measured ``value`` of I near z=1,
        and a known standard deviation ``sigma``.  The rate is linear in the
        mobility (I = mobility * f), so a sigma-weighted least-squares fit of
        the single scalar mobility is exact.
        """
        f = np.array([response(r['input']) for r in records], dtype=float)
        y = np.array([r['value'] for r in records], dtype=float)
        if 'sigma' in records[0]:
            w = 1.0 / np.array([r['sigma'] for r in records], dtype=float) ** 2
        else:
            w = np.ones_like(y)
        denom = np.sum(w * f * f)
        self.mobility = float(np.sum(w * f * y) / denom) if denom > 0 else float('nan')
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
