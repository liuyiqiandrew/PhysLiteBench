"""Average over signed exposure counts of each diffusion path."""
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import cdist


def exposures(scan):
    start = scan["start"]
    counts = [0] * 4
    counts[start] = scan["signs"][0]
    states = {(start, tuple(counts)): 1.0}
    for p, sign in zip(scan["hops"], scan["signs"][1:]):
        after = defaultdict(float)
        for (site, counts), mass in states.items():
            for shift, weight in ((-1, p / 2), (0, 1 - p), (1, p / 2)):
                if weight:
                    dest = (site + shift) % 4
                    updated = list(counts)
                    updated[dest] += sign
                    after[dest, tuple(updated)] += mass * weight
        states = after
    marginal = defaultdict(float)
    for (_, counts), mass in states.items():
        marginal[counts] += mass
    return np.array(list(marginal)), np.array(list(marginal.values()))


def second_moment(scan):
    counts, mass = exposures(scan)
    return mass @ (counts**2).sum(axis=1)


def signal(experiment, width):
    coeff = -0.5 * (experiment["gain"] * width)**2
    counts, mass = exposures(experiment["scan"])
    echo = mass @ np.exp(coeff * (counts**2).sum(axis=1))
    if experiment["mode"] == "echo":
        return echo
    other, weights = exposures(experiment["partner"])
    partner_echo = weights @ np.exp(coeff * (other**2).sum(axis=1))
    joint = mass @ np.exp(coeff * cdist(counts, other, "sqeuclidean")) @ weights
    return joint - echo * partner_echo


class Model:
    def __init__(self):
        self.width = None

    def fit(self, records):
        records = list(records)
        x = np.array([second_moment(row["scan"]) for row in records])
        y = np.array([row["curvature"] for row in records])
        self.width = float(np.sqrt((x @ y) / (x @ x)))
        return self

    def predict_calibration(self, records):
        return np.array([self.width**2 * second_moment(r["scan"]) for r in records])

    def predict(self, experiments):
        return np.array([signal(row, self.width) for row in experiments])
