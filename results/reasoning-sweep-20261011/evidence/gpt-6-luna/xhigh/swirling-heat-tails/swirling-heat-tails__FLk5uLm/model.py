import numpy as np


def response(e):
    """Return the unit-mobility rate for the stationary OU process.

    The circulating-work SCGF is ``1 - sqrt(1 - 4 r^2 lambda(1+lambda))``.
    Total heat differs from that work by the change in trap energy.  The
    stationary endpoint distributions restrict the SCGF domain to
    ``-1 <= lambda <= lambda_max``; the Legendre transform therefore has
    linear tails when its unconstrained saddle lies outside that interval.
    The returned value includes the stiffness time scale; mobility is
    applied separately by ``predict_at``.
    """
    r = float(e['drive_ratio'])
    z = float(e['relative_heat'])
    a = r * r

    # The unconstrained Legendre saddle, written using y=1+2*lambda.
    # This form stays well behaved for both signs of z.
    y = z * np.sqrt((1.0 + a) / (1.0 + a * z * z))
    saddle = 0.5 * (y - 1.0)

    # The negative boundary comes from the final trap-energy endpoint.  On
    # the positive side, the initial endpoint is restrictive only for weak
    # circulation (a < 1/3); otherwise the SCGF square-root edge comes first.
    lower = -1.0
    if a < 1.0 / 3.0:
        upper = (1.0 - a) / (1.0 + a)
    else:
        upper = 0.5 * (np.sqrt(1.0 + 1.0 / a) - 1.0)
    lam = np.clip(saddle, lower, upper)

    radicand = 1.0 - 4.0 * a * lam * (1.0 + lam)
    scgf = 1.0 - np.sqrt(max(0.0, radicand))
    heat_rate = 2.0 * a * z
    return float(e['stiffness']) * (lam * heat_rate - scgf)


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        if not records:
            raise ValueError('At least one calibration record is required.')

        experiments = [record['input'] for record in records]
        basis = np.asarray([response(e) for e in experiments])
        values = np.asarray([float(record['value']) for record in records])
        sigma = np.asarray([float(record['sigma']) for record in records])
        if np.any(~np.isfinite(basis)) or np.any(~np.isfinite(values)):
            raise ValueError('Calibration inputs and values must be finite.')
        if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise ValueError('Calibration standard deviations must be finite and positive.')

        # Independent Gaussian readings give a weighted least-squares fit of
        # the single common mobility.  Project onto its known apparatus range.
        weights = 1.0 / (sigma * sigma)
        denom = np.sum(weights * basis * basis)
        if denom <= 0.0:
            raise ValueError('Calibration records do not identify mobility.')
        fitted = np.sum(weights * basis * values) / denom
        self.mobility = float(np.clip(fitted, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
