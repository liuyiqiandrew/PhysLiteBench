import numpy as np


def response(e):
    """Return the rate function with the mobility factor removed.

    For the linear diffusion in the README, the scaled cumulant generating
    function of the heat is

        lambda(s) = mobility * stiffness *
                    (1 - sqrt(1 - 4*r**2*s*(1+s))).

    The stationary heat rate is ``2 * mobility * stiffness * r**2``.
    Legendre transforming after normalising by that rate gives the expression
    below.  Keeping mobility outside this function makes the common
    calibration parameter explicit.
    """
    r = float(e['drive_ratio'])
    z = float(e['relative_heat'])
    q = 2.0 * r * r * z
    rate = np.sqrt((1.0 + r * r) * (q * q + 4.0 * r * r) /
                   (4.0 * r * r)) - 0.5 * q - 1.0
    # Roundoff can make the exact minimum at z=1 very slightly negative.
    return float(e['stiffness']) * max(0.0, rate)


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        # The readings have independent Gaussian errors, so the maximum
        # likelihood estimate is weighted least squares for y = mobility*x.
        design = np.asarray([response(record['input']) for record in records],
                            dtype=float)
        values = np.asarray([record['value'] for record in records],
                            dtype=float)
        sigma = np.asarray([record['sigma'] for record in records],
                           dtype=float)
        weights = 1.0 / (sigma * sigma)

        numerator = np.sum(weights * design * values)
        denominator = np.sum(weights * design * design)
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError('Calibration records do not identify mobility.')

        estimate = numerator / denominator
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
