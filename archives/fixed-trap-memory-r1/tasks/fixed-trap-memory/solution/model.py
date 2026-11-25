from functools import lru_cache
import numpy as np


@lru_cache(maxsize=2048)
def passage_moments(size, right_probability, residence_ratio, slow_probability, escape_scale):
    jump = np.zeros((size,size))
    jump[0,1] = 1.
    for i in range(1,size):
        jump[i,i-1] = 1-right_probability
        if i+1 < size:
            jump[i,i+1] = right_probability
    green = np.linalg.solve(np.eye(size)-jump, np.eye(size))
    visits = green @ np.ones(size)
    visits_squared = (2*green-np.eye(size)) @ visits
    mean_wait = (1-slow_probability+slow_probability*residence_ratio)/escape_scale
    second_wait_mean = (1-slow_probability+slow_probability*residence_ratio**2)/escape_scale**2
    returns = 2*np.dot(green[0], np.diag(green)-1)
    mean = mean_wait*visits[0]
    second = mean_wait**2*visits_squared[0]+(2*second_wait_mean-mean_wait**2)*visits[0]
    second += (second_wait_mean-mean_wait**2)*returns
    return float(mean), float(second-mean**2)


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
        from scipy.optimize import minimize_scalar
        experiments = [r['input'] for r in records]
        values = np.asarray([r['value'] for r in records])
        sigma = np.asarray([r['sigma'] for r in records])
        result = minimize_scalar(lambda scale: np.sum(((predict_at(experiments,scale)-values)/sigma)**2),
                                 bounds=(.6,1.4), method='bounded', options={'xatol':1e-12})
        self.escape_scale = float(result.x)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.escape_scale)
