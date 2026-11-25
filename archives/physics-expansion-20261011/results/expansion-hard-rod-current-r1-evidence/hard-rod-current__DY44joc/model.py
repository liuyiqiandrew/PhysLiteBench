import numpy as np


def response(e):
    """Return the current response with unit particle speed.

    The hard-rod equilibrium can be represented by exponential free gaps,
    with the rod length removed from each gap.  In that representation the
    current SCGF is a drift plus two compound-Poisson terms.  The speed is a
    separate overall scale, so this function uses speed one.
    """
    rho, a, p = e['density'], e['length'], e['right_fraction']
    h, d = 1-a*rho, 2*p-1
    if e['statistic'] == 'mean':
        return rho*d
    if e['statistic'] == 'variance':
        return rho*h*(1-a*rho*d*d)
    field = e['counting_field']
    c = rho/h
    plus = c*p*(1-a*rho*d)
    minus = c*(1-p)*(1+a*rho*d)
    return (rho*d*field + plus*(np.expm1(h*field)-h*field)
            + minus*(np.expm1(-h*field)+h*field))


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        # Every observable is linear in the common speed.  Use all records
        # together and account for their quoted Gaussian uncertainties.
        x = np.asarray([response(record['input']) for record in records],
                       dtype=float)
        y = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records],
                           dtype=float)

        weights = 1.0/(sigma*sigma)
        denominator = np.sum(weights*x*x)
        if denominator == 0.0:
            raise ValueError('Calibration contains no informative records.')

        speed = np.sum(weights*x*y)/denominator
        # The apparatus specifies the admissible speed interval.  Clipping
        # also keeps a noisy calibration from producing an out-of-domain
        # model when the fit is used on new experiments.
        self.speed = float(np.clip(speed, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
