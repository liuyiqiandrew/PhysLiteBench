import numpy as np


def response(e):
    """Rate function per unit mobility for one preparation.

    With ``a = drive_ratio**2``, the heat rate is
    ``q_bar = 2 * mobility * stiffness * a``.  Solving the tilted
    Ornstein--Uhlenbeck problem gives, for Q/(q_bar*t)=z,

        mobility * stiffness * [sqrt((1+a)(1+a*z**2)) - 1 - a*z].

    The quotient below is the same expression after rationalising the
    difference.  It is important for calibration points close to z=1,
    where the unsubtracted expression loses most of its precision.
    """
    drive_ratio = float(e['drive_ratio'])
    relative_heat = float(e['relative_heat'])
    stiffness = float(e['stiffness'])

    a = drive_ratio * drive_ratio
    root = np.sqrt((1.0 + a) * (1.0 + a * relative_heat * relative_heat))
    dimensionless_rate = a * (relative_heat - 1.0) ** 2 / (
        root + 1.0 + a * relative_heat
    )
    return stiffness * dimensionless_rate


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        """Fit the common mobility by Gaussian weighted least squares."""
        if not records:
            raise ValueError('at least one calibration record is required')

        base_rate = np.asarray([response(record['input']) for record in records])
        observed_rate = np.asarray([float(record['value']) for record in records])
        sigma = np.asarray([float(record['sigma']) for record in records])

        if (not np.isfinite(base_rate).all() or not np.isfinite(observed_rate).all()
                or not np.isfinite(sigma).all() or np.any(sigma <= 0.0)):
            raise ValueError('calibration values and standard deviations must be finite, with sigma > 0')

        weights = 1.0 / (sigma * sigma)
        denominator = np.sum(weights * base_rate * base_rate)
        if denominator == 0.0:
            raise ValueError('calibration inputs do not identify mobility')

        estimate = np.sum(weights * base_rate * observed_rate) / denominator
        # Mobility is specified by the apparatus to be in this closed interval.
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
