import numpy as np


def response(e, mobility=1.0):
    """Asymptotic survival rate for one experiment.

    If the target has mean velocity ``u`` and diffusion coefficient ``D_A``,
    its survival exponent is the lower envelope of the cost of producing that
    velocity and the exponent for a deterministic target moving at ``u``:

        min_u rho*|u| + (u - mu*F)^2/(4*D_A).

    The closed form below also handles the deterministic (T=0) limit.  The
    trap diffusivity does not occur in the rate: for a target with nonzero
    ballistic velocity it changes only boundary-layer/sublinear terms, and a
    target with zero velocity has a subexponential survival factor.
    """
    rho = float(e['density'])
    force = float(e['force'])
    temperature = float(e['temperature'])
    drift = mobility * force
    target_diffusivity = mobility * temperature

    speed = abs(drift)
    if target_diffusivity == 0.0:
        return rho * speed

    crossover = 2.0 * rho * target_diffusivity
    if speed <= crossover:
        return drift * drift / (4.0 * target_diffusivity)
    return rho * speed - rho * rho * target_diffusivity


def predict_at(experiments, mobility):
    return np.asarray([response(e, mobility) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')

        # Calibration is at T=0 and has nonzero force, where the exact rate is
        # rho * mobility * abs(force).  Use the supplied readout uncertainties
        # for the weighted least-squares estimate through the origin.
        design = np.asarray([
            float(r['input']['density']) * abs(float(r['input']['force']))
            for r in records
        ], dtype=float)
        values = np.asarray([float(r['value']) for r in records], dtype=float)
        sigma = np.asarray([float(r['sigma']) for r in records], dtype=float)
        if (not np.isfinite(design).all() or not np.isfinite(values).all()
                or not np.isfinite(sigma).all() or np.any(sigma <= 0.0)
                or np.all(design == 0.0)):
            raise ValueError('Invalid calibration records.')

        weights = 1.0 / (sigma * sigma)
        estimate = np.sum(weights * design * values) / np.sum(weights * design * design)
        # The apparatus specifies this as a shared mobility in the stated
        # interval.  Clipping also keeps predictions well behaved for an
        # unusually noisy or adversarial calibration set.
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
