import numpy as np
from scipy.special import expit


def transition_rates(e, rate):
    fraction = float(e['fraction'])
    voltage = np.array([e['voltage_left'], e['voltage_right']])
    epsilon = e['offset'] - fraction * voltage[0] - (1 - fraction) * voltage[1]
    bare = rate * np.array([e['left_factor'], e['right_factor']])
    occupation = expit(-(epsilon + voltage) / e['thermal_energy'])
    return bare * occupation, bare * (1 - occupation)


def spectrum(e, rate):
    entry, exit = transition_rates(e, rate)
    a, b = entry.sum(), exit.sum()
    p = np.array([b, a]) / (a + b)
    generator = np.array([[-a, b], [a, -b]])
    weights = np.array([1 - e['fraction'], -e['fraction']])
    current = np.array([[0., -weights @ exit], [weights @ entry, 0.]])
    square = np.array([[0., weights**2 @ exit], [weights**2 @ entry, 0.]])
    stationary = np.outer(p, np.ones(2))
    resolvent = np.linalg.solve(1j * e['omega'] * np.eye(2) - generator + stationary,
                               np.eye(2) - stationary)
    white = 2 * np.ones(2) @ square @ p
    correlation = 4 * np.real(np.ones(2) @ current @ resolvent @ current @ p)
    return float(white + correlation)


def predict_at(experiments, rate):
    return np.asarray([spectrum(e, rate) for e in experiments])


class Model:
    def __init__(self):
        self.rate = None

    def fit(self, records):
        records = list(records)
        basis = predict_at([r['input'] for r in records], 1.0)
        y = np.asarray([r['value'] for r in records])
        sigma = np.asarray([r['sigma'] for r in records])
        self.rate = float(np.clip(np.dot(basis / sigma, y / sigma)
                                 / np.dot(basis / sigma, basis / sigma), .7, 1.4))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.rate)
