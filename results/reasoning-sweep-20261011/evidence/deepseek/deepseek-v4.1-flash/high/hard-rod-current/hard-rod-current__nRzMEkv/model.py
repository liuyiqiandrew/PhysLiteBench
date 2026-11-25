import numpy as np
from scipy.optimize import brentq


def response(e):
    """Unit-speed rod-current response for one apparatus setting.

    The hard-rod gas maps onto independent carriers on the free length
    ``1/(1 - density*length)`` per unit length, so all three readouts depend on
    the *mean* rod length (the length order / contrast drops out in the
    thermodynamic limit).  The common speed fixes the time scale and enters
    every readout linearly, so the response is tabulated at unit speed and
    rescaled by the fitted speed.
    """
    rho, mean = e['density'], e['length']
    if e['statistic'] == 'mean':
        return 0.
    if e['statistic'] == 'variance':
        return rho*(1-rho*mean)
    field = abs(e['counting_field'])
    c = rho/(1-rho*mean)
    s = brentq(lambda s: s+c*mean*np.sinh(s)-field, 0., field+1, xtol=1e-14)
    return c*2*np.sinh(s/2)**2


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        """Weighted least-squares (Gaussian MLE) fit of the common speed.

        Each record is ``input``/``value``/``sigma`` with independent Gaussian
        noise, so the maximum-likelihood speed minimizes
        ``sum_i (value_i - speed*response_i)^2 / sigma_i^2``.  The mean readout
        has zero response and carries no information about the speed.
        """
        x = np.asarray([response(r['input']) for r in records], dtype=float)
        y = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        w = 1.0/sigma**2
        denom = float(np.sum(w*x*x))
        self.speed = float(np.sum(w*x*y)/denom) if denom > 0 else 1.0
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
