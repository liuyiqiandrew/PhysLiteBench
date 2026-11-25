import numpy as np


def predict_at(experiments, mobility):
    """Return the asymptotic winding-readout rate for each experiment.

    For the isotropic Ornstein--Uhlenbeck process, conditioning on the
    radial path makes the angular increment a time-changed Brownian motion.
    The corresponding radial Feynman--Kac ground state is ``r**abs(lambda)``
    and has eigenvalue ``-mobility * stiffness * abs(lambda)``.
    """
    scale = np.asarray(
        [e['stiffness'] * abs(e['probe']) for e in experiments], dtype=float
    )
    return np.asarray(mobility * scale, dtype=float)


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        """Estimate the shared mobility by weighted least squares.

        The calibration records have the linear form
        ``value = mobility * stiffness * abs(probe) + noise``.  Using the
        supplied readout standard deviations gives the Gaussian maximum
        likelihood estimate, including if future calibration sets use
        unequal uncertainties.
        """
        scale = np.asarray(
            [r['input']['stiffness'] * abs(r['input']['probe']) for r in records],
            dtype=float,
        )
        value = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)

        weights = 1.0 / sigma**2
        self.mobility = float(np.sum(weights * scale * value) /
                              np.sum(weights * scale**2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
