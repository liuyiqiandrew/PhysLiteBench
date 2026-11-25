import numpy as np


def response(e):
    """Return the rate function with the mobility factor removed.

    Rescaling time by ``mobility * stiffness`` gives an OU process with
    mean heat rate ``2 * drive_ratio**2``.  Its interior rate function is
    obtained by Legendre transforming

        lambda(s) = 1 - sqrt(1 - 4*r**2*s*(1+s)).

    The heat includes the trap-energy endpoint term.  Since the endpoints
    are sampled from the stationary state rather than fixed, that term
    changes the two far tails of the rate function.  The resulting rate is
    linear for z < -1, and also for sufficiently large positive z when
    r < 1/sqrt(3).
    """
    r = float(e['drive_ratio'])
    z = float(e['relative_heat'])

    # The dimensionless (mobility * stiffness = 1) heat rate is
    # w = 2*r**2*z.
    if z < -1.0:
        rate = -2.0 * r * r * z
    else:
        # The upper end of the SCGF domain is set by the area singularity,
        # except at small r where the stationary initial endpoint dominates.
        if r * r < 1.0 / 3.0:
            z_upper = (3.0 - r * r) / (1.0 - 3.0 * r * r)
        else:
            z_upper = np.inf

        if z > z_upper:
            # At this boundary s=(1-r**2)/(1+r**2), and
            # lambda(s)=4*r**2/(1+r**2).
            rate = (
                2.0 * r * r * (1.0 - r * r) / (1.0 + r * r) * z
                - 4.0 * r * r / (1.0 + r * r)
            )
        else:
            q = 2.0 * r * r * z
            rate = np.sqrt((1.0 + r * r) * (q * q + 4.0 * r * r) /
                           (4.0 * r * r)) - q / 2.0 - 1.0

    return float(e['stiffness']) * rate


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        if not records:
            raise ValueError('At least one calibration record is required.')

        x = np.asarray([response(record['input']) for record in records],
                       dtype=float)
        y = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)

        if (not np.isfinite(x).all() or not np.isfinite(y).all() or
                not np.isfinite(sigma).all() or np.any(sigma <= 0.0)):
            raise ValueError('Calibration values and uncertainties must be finite, '
                             'with positive sigma.')

        # Independent Gaussian readings make weighted least squares the
        # maximum-likelihood estimate for the common multiplicative mobility.
        weights = 1.0 / (sigma * sigma)
        denominator = np.sum(weights * x * x)
        if denominator <= 0.0:
            raise ValueError('Calibration records do not identify mobility.')
        estimate = np.sum(weights * x * y) / denominator
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
