import numpy as np
from scipy.special import jv
from scipy.optimize import minimize_scalar


def response(e, gamma, cutoff=64):
    """Return the requested Fourier component for one experiment.

    The calculation uses the characteristic function of the stationary
    Ornstein--Uhlenbeck displacement and a Jacobi--Anger expansion for each
    velocity impulse.  The integer selection rule from the initially uniform
    position makes the resulting sum one-dimensional.
    """
    tau, u = e['delay'], e['after']
    speed2 = float(e['thermal_speed']) ** 2

    # A(t) is the displacement response to a unit velocity kick.  V(t) is
    # the variance of the equilibrium OU displacement over an interval t.
    a = lambda t: -np.expm1(-gamma * t) / gamma
    v = lambda t: 2 * speed2 * (gamma * t + np.expm1(-gamma * t)) / gamma**2

    k1, k2, m = e['k1'], e['k2'], e['mode']
    n = np.arange(-cutoff, cutoff+1)
    keep = (m+n*k2) % k1 == 0
    n = n[keep]
    ell = -(m+n*k2)//k1
    q = m+n*k2

    # The two free displacements share the OU velocity around the second
    # impulse.  Stationarity gives Cov(Y_tau, Y_u) = s^2 A(tau) A(u).
    cov = speed2 * a(tau) * a(u)
    variance = q*q*v(tau)+m*m*v(u)+2*q*m*cov
    terms = jv(n,m*e['a2']*a(u))*jv(ell,e['a1']*(m*a(tau+u)+n*k2*a(tau)))
    terms = terms*np.exp(-variance/2+1j*(n*e['phase2']+ell*e['phase1']))
    value = terms.sum()
    return float(value.real if e['component']=='cos' else value.imag)


def predict_at(experiments, friction):
    return np.asarray([response(e, float(friction)) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.friction = None

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError("records must not be empty")

        # The calibration uncertainties are independent and supplied with
        # every record, so the natural objective is weighted chi-squared.
        def objective(gamma):
            residuals = np.array([
                (response(record['input'], gamma) - record['value']) /
                record['sigma']
                for record in records
            ], dtype=float)
            return float(np.dot(residuals, residuals))

        result = minimize_scalar(objective, bounds=(0.08, 0.4), method='bounded')
        if not result.success or not np.isfinite(result.x):
            raise RuntimeError("could not fit friction")
        self.friction = float(result.x)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.friction)
