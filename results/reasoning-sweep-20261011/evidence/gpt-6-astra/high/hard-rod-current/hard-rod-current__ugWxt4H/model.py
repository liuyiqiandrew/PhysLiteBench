import numpy as np
from scipy.optimize import brentq


def response(e):
    """Long-time readout at unit speed, averaged over the rod order.

    Removing the rod lengths leaves a Poisson gas of density c. For a
    positive current, the lengths of the crossed rods enter through their
    Laplace transform M(k) = E[exp(-k*A)]. The annealed saddle satisfies
    field = s - log(M(c*sinh(s))), and its SCGF is c*(cosh(s)-1).
    Reflection symmetry supplies the negative-field branch.
    """
    rho, mean = e['density'], e['length']
    if e['statistic'] == 'mean':
        return 0.
    if e['statistic'] == 'variance':
        # Only E[A] enters the quadratic term, even for a length mixture.
        return rho*(1-rho*mean)
    if e['statistic'] != 'scgf':
        raise ValueError("Unknown statistic: {!r}".format(e['statistic']))
    field = abs(e['counting_field'])
    if field == 0.:
        return 0.
    c = rho/(1-rho*mean)
    delta = mean*e['contrast']
    shortest = mean-delta

    def saddle(s):
        k = c*np.sinh(s)
        # -log M(k) = k*shortest - log((1 + exp(-2*k*delta))/2).
        # expm1/log1p preserve accuracy near zero and avoid overflow.
        return s+k*shortest-np.log1p(0.5*np.expm1(-2*k*delta))-field

    # Positive lengths make the saddle equation strictly increasing, with
    # its unique root between zero and field. Relative tolerance also keeps
    # predictions accurate for fields much smaller than machine epsilon.
    s = brentq(saddle, 0., field, xtol=np.nextafter(0., 1.), rtol=1e-14)
    return c*2*np.sinh(s/2)**2


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        """Fit the shared speed by Gaussian maximum likelihood."""
        records = list(records)
        unit = np.asarray([response(r['input']) for r in records], dtype=float)
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        if not (np.isfinite(values).all() and np.isfinite(sigma).all()
                and (sigma > 0).all()):
            raise ValueError('Calibration values must be finite and sigma positive.')
        weighted_unit = unit/sigma
        information = np.dot(weighted_unit, weighted_unit)
        if information == 0.:
            raise ValueError('Calibration must contain a nonzero speed response.')
        estimate = np.dot(weighted_unit, values/sigma)/information
        self.speed = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.speed is None:
            raise ValueError('Fit the model before predicting.')
        return predict_at(experiments, self.speed)
