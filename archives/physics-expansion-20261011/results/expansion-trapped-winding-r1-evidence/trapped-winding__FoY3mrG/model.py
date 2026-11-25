import numpy as np


def predict_at(experiments, mobility):
    """Return the asymptotic winding-decay rate for each experiment.

    For the isotropic OU process, the radial eigenfunction associated with
    an angular probe ``lambda`` has power ``abs(lambda)``.  Its eigenvalue is
    therefore ``-mobility * stiffness * abs(lambda)``.  Temperature sets the
    radial scale but cancels from this rate.
    """
    values = np.asarray(
        [e['stiffness'] * abs(e['probe']) for e in experiments],
        dtype=float,
    )
    return mobility * values


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        # The calibration noise is additive with known standard deviation, so
        # use the corresponding inverse-variance weighted least-squares fit
        # through the origin for rate = mobility * k * |lambda|.
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')

        x = np.asarray(
            [r['input']['stiffness'] * abs(r['input']['probe']) for r in records],
            dtype=float,
        )
        y = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)

        weights = 1.0 / np.square(sigma)
        denominator = np.sum(weights * np.square(x))
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError('Calibration records do not identify mobility.')

        self.mobility = float(np.sum(weights * x * y) / denominator)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
