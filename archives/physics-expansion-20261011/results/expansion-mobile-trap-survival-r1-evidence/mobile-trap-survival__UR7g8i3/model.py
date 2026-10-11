import numpy as np


def response(e, trap_diffusivity):
    """Asymptotic readout for mobile Poisson traps on an infinite line.

    At the leading order in ``sqrt(t)``, the target diffusivity only changes
    subleading terms.  The target-survival coefficient is therefore the same
    as for a stationary target:

        -log(S(t)) / sqrt(t) -> 4 * rho * sqrt(D_B / pi).
    """
    return 4 * e['density'] * np.sqrt(trap_diffusivity / np.pi)


def predict_at(experiments, trap_diffusivity):
    return np.asarray([response(e, trap_diffusivity) for e in experiments])


class Model:
    def __init__(self):
        self.trap_diffusivity = None

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError('at least one calibration record is required')

        density = np.asarray([r['input']['density'] for r in records], dtype=float)
        value = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)

        # The response is C = slope * density, where slope = 4 sqrt(DB/pi).
        # Use the supplied readout uncertainties in the least-squares fit.
        weight = 1.0 / np.square(sigma)
        slope = np.sum(weight * density * value) / np.sum(weight * density**2)
        diffusivity = np.pi * np.square(slope / 4.0)

        # The apparatus specifies the admissible range for the shared value.
        self.trap_diffusivity = float(np.clip(diffusivity, 0.5, 1.5))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.trap_diffusivity)
