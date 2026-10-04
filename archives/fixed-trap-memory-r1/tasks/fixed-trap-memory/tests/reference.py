"""Explicit binary specimens and conditional continuous-time hitting moments."""
from functools import lru_cache
import itertools
import numpy as np

TRUE_PARAMETER = 1.07


@lru_cache(maxsize=2048)
def passage_moments(size, right_probability, residence_ratio, slow_probability, escape_scale):
    averaged_first = 0.
    averaged_second = 0.
    for configuration in itertools.product([0,1], repeat=size):
        number_slow = sum(configuration)
        weight = slow_probability**number_slow*(1-slow_probability)**(size-number_slow)
        residence = np.where(np.asarray(configuration) == 0, 1., residence_ratio)/escape_scale
        backward = np.diag(1/residence)
        backward[0,1] = -1/residence[0]
        for site in range(1,size):
            backward[site,site-1] = -(1-right_probability)/residence[site]
            if site+1 < size:
                backward[site,site+1] = -right_probability/residence[site]
        first = np.linalg.solve(backward, np.ones(size))
        second = np.linalg.solve(backward, 2*first)
        averaged_first += weight*first[0]
        averaged_second += weight*second[0]
    return float(averaged_first), float(averaged_second-averaged_first**2)


def predict(experiments, escape_scale):
    output = []
    for e in experiments:
        pair = passage_moments(e['size'],e['right_probability'],e['residence_ratio'],e['slow_probability'],escape_scale)
        output.append(pair[0 if e['statistic'] == 'mean' else 1])
    return np.asarray(output)


def instrument_sigma(e):
    return .1 if e['statistic'] == 'mean' else .5


def experiment(size, right_probability, residence_ratio, slow_probability, statistic):
    return dict(size=size,right_probability=right_probability,residence_ratio=residence_ratio,
                slow_probability=slow_probability,statistic=statistic)


def calibration_inputs():
    means = [experiment(n,p,c,s,'mean') for n in [3,5,7] for p in [.5,.7,.9] for c,s in [(3.,.25),(10.,.6)]]
    variances = [experiment(n,p,1.,.4,'variance') for n in [3,5] for p in [.55,.85]]
    return means*12+variances*18


def hidden_inputs():
    def variances(items):return [experiment(*x,'variance') for x in items]
    return {
        'short_channels':variances([(3,.5,4.,.5),(3,.5,12.,.3),(3,.5,25.,.15),(5,.55,12.,.3)]),
        'long_channels':variances([(7,.65,4.,.5),(7,.65,12.,.3),(7,.65,25.,.15),(9,.75,25.,.15)]),
        'mixed_channels':variances([(5,.55,4.,.5),(5,.55,25.,.15),(9,.75,4.,.5),(9,.75,12.,.3)]),
        'exact_anchors':[experiment(8,.65,9.,.35,'mean'),experiment(6,.9,20.,.7,'mean'),
                         experiment(4,1.,12.,.3,'variance'),experiment(8,.6,1.,.8,'variance')]
    }
