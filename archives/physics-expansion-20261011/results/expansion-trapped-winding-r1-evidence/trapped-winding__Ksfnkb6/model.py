import numpy as np


def predict_at(experiments, mobility):
    """Evaluate the exact asymptotic winding rate.

    For this isotropic Ornstein-Uhlenbeck process, the angular winding has a
    Cauchy-type long-time characteristic function.  Its decay rate is linear
    in the magnitude of the angular probe (and independent of temperature):

        rate = mobility * stiffness * abs(probe).
    """
    return mobility * np.asarray(
        [e['stiffness'] * abs(e['probe']) for e in experiments], dtype=float
    )


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        # The calibration readings have independent Gaussian errors, so use
        # the inverse-variance weighted least-squares estimate for the single
        # shared mobility parameter.  The known physical interval also keeps
        # the estimate physical for unusually noisy/small calibration sets.
        design = np.asarray(
            [r['input']['stiffness'] * abs(r['input']['probe']) for r in records],
            dtype=float,
        )
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)

        weights = 1.0 / sigma**2
        denominator = np.sum(weights * design**2)
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError('Calibration records do not identify mobility.')

        estimate = np.sum(weights * design * values) / denominator
        self.mobility = float(np.clip(estimate, 0.7, 1.4))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
