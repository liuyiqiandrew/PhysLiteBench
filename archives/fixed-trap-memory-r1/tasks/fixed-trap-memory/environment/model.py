from functools import lru_cache
import numpy as np


@lru_cache(maxsize=2048)
def passage_moments(size, right_probability, residence_ratio, slow_probability, escape_scale):
    probability = np.array([1-slow_probability, slow_probability])
    rates = escape_scale/np.array([1., residence_ratio])
    transition = np.zeros((2*size, 2*size))
    for site in range(size):
        neighbors = [(1, 1.)] if site == 0 else [(site-1, 1-right_probability), (site+1, right_probability)]
        for kind in range(2):
            row = 2*site+kind
            transition[row,row] = -rates[kind]
            for destination, direction_probability in neighbors:
                if destination < size:
                    transition[row,2*destination:2*destination+2] += rates[kind]*direction_probability*probability
    first = np.linalg.solve(-transition, np.ones(2*size))
    second = np.linalg.solve(-transition, 2*first)
    mean = probability @ first[:2]
    variance = probability @ second[:2] - mean**2
    return float(mean), float(variance)


def predict_at(experiments, escape_scale):
    result = []
    for experiment in experiments:
        moments = passage_moments(int(experiment['size']), float(experiment['right_probability']),
                                  float(experiment['residence_ratio']), float(experiment['slow_probability']),
                                  float(escape_scale))
        result.append(moments[0 if experiment['statistic'] == 'mean' else 1])
    return np.asarray(result, dtype=float)


class Model:
    def __init__(self):
        self.escape_scale = None

    def fit(self, records):
        raise NotImplementedError('Fit the common escape-rate scale from the calibration records.')

    def predict(self, experiments):
        return predict_at(experiments, self.escape_scale)
