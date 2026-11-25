"""Sequential-tunnelling model for the two-state island."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit


RATE_MIN = 0.7
RATE_MAX = 1.4


def transition_rates(e, rate):
    """Return entry and exit rates for the left and right junctions."""
    fraction = float(e['fraction'])
    voltage = np.array([e['voltage_left'], e['voltage_right']], dtype=float)
    epsilon = (float(e['offset']) - fraction * voltage[0]
               - (1.0 - fraction) * voltage[1])
    bare = float(rate) * np.array([e['left_factor'], e['right_factor']],
                                  dtype=float)
    occupation = expit(-(epsilon + voltage) / float(e['thermal_energy']))
    return bare * occupation, bare * (1.0 - occupation)


def spectrum(e, rate):
    """Return the connected source-current spectrum for one experiment.

    The measured source wire carries both the direct tunnelling current and
    the instantaneous capacitive displacement current.  If f is the left
    capacitance fraction, the four event charges (left entry, left exit,
    right entry, right exit) are 1-f, -(1-f), -f, f.
    """
    entry, exit = transition_rates(e, rate)
    a, b = float(entry.sum()), float(exit.sum())
    stationary = np.array([b, a], dtype=float) / (a + b)
    ones = np.ones(2, dtype=float)
    projector = np.outer(stationary, ones)
    generator = np.array([[-a, b], [a, -b]], dtype=float)

    fraction = float(e['fraction'])
    charges = np.array([1.0 - fraction, -(1.0 - fraction),
                        -fraction, fraction], dtype=float)
    events = (
        np.array([[0., 0.], [entry[0], 0.]]),
        np.array([[0., exit[0]], [0., 0.]]),
        np.array([[0., 0.], [entry[1], 0.]]),
        np.array([[0., exit[1]], [0., 0.]]),
    )
    jump = sum((q * event for q, event in zip(charges, events)),
               start=np.zeros((2, 2)))
    jump_square = sum((q * q * event for q, event in zip(charges, events)),
                      start=np.zeros((2, 2)))

    # Integral of the decaying propagator for positive time.  The projector
    # removes its stationary component.
    resolvent = np.linalg.solve(
        -1j * float(e['omega']) * np.eye(2) - generator + projector,
        np.eye(2) - projector,
    )
    white = 2.0 * ones @ jump_square @ stationary
    correlation = 4.0 * np.real(ones @ jump @ resolvent @ jump @ stationary)
    return float(white + correlation)


def predict_at(experiments, rate):
    return np.asarray([spectrum(e, rate) for e in experiments])


class Model:
    def __init__(self):
        self.rate = None

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError('fit requires at least one calibration record')
        experiments = [r['input'] for r in records]
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        if (not np.isfinite(values).all() or not np.isfinite(sigma).all()
                or np.any(sigma <= 0)):
            raise ValueError('calibration values and sigmas must be finite, positive data')

        def objective(rate):
            residual = (predict_at(experiments, rate) - values) / sigma
            return float(residual @ residual)

        result = minimize_scalar(objective, bounds=(RATE_MIN, RATE_MAX),
                                 method='bounded', options={'xatol': 1e-12})
        candidates = [(float(result.fun), float(result.x))]
        candidates.extend((objective(bound), bound)
                          for bound in (RATE_MIN, RATE_MAX))
        _, self.rate = min(candidates, key=lambda item: item[0])
        return self

    def predict(self, experiments):
        rate = RATE_MIN if self.rate is None else float(self.rate)
        return predict_at(experiments, rate)
