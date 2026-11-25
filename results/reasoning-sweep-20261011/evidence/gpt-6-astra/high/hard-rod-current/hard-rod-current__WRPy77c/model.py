import numpy as np
from scipy.optimize import brentq


def response(e):
    """Long-time readout at unit speed, with annealed rod lengths.

    Removing occupied lengths gives a Poisson gas of density c. For a
    positive current j, the removed length is the sum of j*t iid lengths.
    Averaging this sum before taking the logarithm gives the saddle equation

        |field| = s - log E[exp(-c*A*sinh(s))],

    where A is equally likely to be either rod length. The SCGF at this
    saddle is c*(cosh(s)-1). Reflection gives the same answer at -field.
    Only E[A] enters the curvature at zero, so contrast does not affect
    the variance, although it does affect the full generating function.
    """
    rho, mean = e['density'], e['length']
    if e['statistic'] == 'mean':
        return 0.
    if e['statistic'] == 'variance':
        return rho*(1-rho*mean)
    if e['statistic'] != 'scgf':
        raise ValueError("Unknown statistic: {!r}".format(e['statistic']))
    field = abs(e['counting_field'])
    if field == 0.:
        return 0.
    c = rho/(1-rho*mean)
    half_difference = mean*e['contrast']

    def saddle(s):
        displacement_tilt = c*np.sinh(s)
        z = half_difference*displacement_tilt
        # log(cosh(z)), accurate near zero and safe throughout the domain.
        if z < 1.:
            log_cosh = np.log1p(2*np.sinh(z/2)**2)
        else:
            log_cosh = z + np.log1p(np.exp(-2*z)) - np.log(2.)
        return s + mean*displacement_tilt - log_cosh - field

    # Positive lengths make the saddle function strictly increasing; its
    # unique root lies between zero and |field|. A tiny absolute tolerance
    # also preserves the quadratic limit for very small counting fields.
    s = brentq(saddle, 0., field, xtol=np.finfo(float).tiny,
               rtol=4*np.finfo(float).eps)
    return c*2*np.sinh(s/2)**2


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        records = list(records)
        basis = np.asarray([response(r['input'])/r['sigma'] for r in records])
        observed = np.asarray([r['value']/r['sigma'] for r in records])
        information = np.dot(basis, basis)
        if information == 0.:
            raise ValueError('Calibration must contain a nonzero response.')
        # All rates scale linearly with speed. This is the Gaussian MLE,
        # constrained to the apparatus's known speed interval.
        self.speed = float(np.clip(np.dot(basis, observed)/information, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.speed is None:
            raise ValueError('Fit the model before predicting.')
        return predict_at(experiments, self.speed)
