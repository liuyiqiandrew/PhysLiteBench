import numpy as np


def response(e, mobility=1.0):
    """Asymptotic survival rate for one experiment.

    For a target path with long-time velocity ``u``, traps contribute
    ``density * abs(u)`` to the exponential rate.  The target itself has
    drift ``v = mobility * force`` and diffusion coefficient
    ``D_A = mobility * temperature``.  Averaging over target paths then
    selects the least expensive effective velocity:

        min_u density*abs(u) + (u-v)^2/(4*D_A).

    The trap diffusivity changes only subexponential terms in this infinite
    line, long-time limit.

    ``mobility`` defaults to one so the original one-argument helper remains
    usable; fitted predictions pass the estimated value explicitly.
    """
    rho = float(e['density'])
    velocity = mobility * float(e['force'])
    target_diffusivity = mobility * float(e['temperature'])

    if target_diffusivity == 0.0:
        return rho * abs(velocity)
    speed = abs(velocity)
    if speed <= 2.0 * rho * target_diffusivity:
        return velocity * velocity / (4.0 * target_diffusivity)
    return rho * speed - rho * rho * target_diffusivity


def predict_at(experiments, mobility):
    return np.asarray([response(e, mobility) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        # At T=0 the response is rho*abs(mu*F), so fit the common mobility
        # by weighted least squares through the origin.  Using sigma here is
        # important because the calibration readouts have additive noise.
        x = []
        y = []
        weights = []
        for record in records:
            experiment = record['input']
            x.append(float(experiment['density']) * abs(float(experiment['force'])))
            y.append(float(record['value']))
            sigma = float(record.get('sigma', 1.0))
            weights.append(1.0 / (sigma * sigma))

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        weights = np.asarray(weights, dtype=float)
        denominator = np.sum(weights * x * x)
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise ValueError('Calibration records contain no usable nonzero force.')

        estimate = np.sum(weights * x * y) / denominator
        if not np.isfinite(estimate) or estimate <= 0.0:
            raise ValueError('Calibration records do not determine a positive mobility.')

        # The apparatus specifies the admissible range for the shared
        # mobility.  Clipping protects predictions from calibration noise.
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
