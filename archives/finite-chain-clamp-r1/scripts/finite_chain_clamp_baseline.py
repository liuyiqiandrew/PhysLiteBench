import numpy as np
from scipy.optimize import brentq


def langevin(z):
    if abs(z) < 1e-3:
        return z/3 - z**3/45 + 2*z**5/945
    return 1/np.tanh(z) - 1/z


def mean_extension(links, length, temperature, force):
    return links*length*langevin(force*length/temperature)


def holding_force(links, length, temperature, extension):
    ratio = extension/(links*length)
    z = brentq(lambda z: langevin(z)-ratio, 0., 2/(1-ratio), xtol=1e-13)
    return temperature*z/length


def predict_at(experiments, length):
    values = []
    for e in experiments:
        if e['mode'] == 'force':
            value = mean_extension(e['links'], length, e['temperature'], e['force'])
        else:
            value = holding_force(e['links'], length, e['temperature'], e['extension'])
        values.append(value)
    return np.asarray(values, dtype=float)


class Model:
    def __init__(self):
        self.link_length = None

    def fit(self, records):
        from scipy.optimize import minimize_scalar
        experiments = [r['input'] for r in records]
        observed = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])

        def loss(length):
            return float(np.sum(((predict_at(experiments, length)-observed)/sigma)**2))

        result = minimize_scalar(loss, bounds=(.8, 1.2), method='bounded',
                                 options={'xatol': 1e-12})
        self.link_length = min([.8, float(result.x), 1.2], key=loss)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.link_length)
