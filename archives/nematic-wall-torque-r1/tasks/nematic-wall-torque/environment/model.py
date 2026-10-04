"""Nematic anchoring response."""
from functools import lru_cache
import numpy as np
from scipy.integrate import solve_bvp


@lru_cache(maxsize=512)
def radial_solution(inner_angle, outer_angle, radius_ratio):
    length = np.log(radius_ratio)
    x = np.linspace(0, 1, 40)
    y = np.vstack((inner_angle + (outer_angle-inner_angle)*x,
                   np.full_like(x, outer_angle-inner_angle)))
    result = solve_bvp(
        lambda x, y: np.vstack((y[1], length**2*np.sin(y[0])*np.cos(y[0]))),
        lambda left, right: np.array([left[0]-inner_angle, right[0]-outer_angle]),
        x, y, tol=1e-9, max_nodes=2048,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.y[1, -1]/length)


def wall_response(experiment):
    inner, outer = experiment['inner_angle'], experiment['outer_angle']
    if experiment['geometry'] == 'flat':
        return (outer-inner)/experiment['thickness']
    ratio = experiment['radius_ratio']
    radius = experiment['inner_radius']*ratio
    slope = radial_solution(inner, outer, ratio)/radius
    return slope + np.sin(outer)*np.cos(outer)/radius


def predict_at(experiments, elastic_constant):
    return elastic_constant*np.array([wall_response(e) for e in experiments])


class Model:
    def __init__(self):
        self.elastic_constant = None

    def fit(self, records):
        raise NotImplementedError('Fit elastic_constant from the calibration records.')

    def predict(self, experiments):
        if self.elastic_constant is None:
            raise RuntimeError('Call fit before predict.')
        return predict_at(experiments, self.elastic_constant)
