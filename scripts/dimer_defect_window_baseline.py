"""Keep the exact boundary-partner marginals but discard their correlations."""
import numpy as np

from boundary import folded_average, pairings
from contours import reading
from model_api import Detector
from ports import distribution


class Model(Detector):
    def loop(self, q):
        if q["mode"] != "fold":
            return reading(q, self.activity)
        n = len(q["ports"])
        patterns = pairings(0, n)
        plates = []
        for plate in q["plates"]:
            exact = distribution(q, self.activity, plate)
            marginal = np.zeros((n, n))
            for pattern, weight in zip(patterns, exact):
                for a, b in pattern:
                    marginal[a, b] += weight
            product = np.array(
                [np.prod([marginal[a, b] for a, b in p]) for p in patterns]
            )
            plates.append(product / product.sum())
        return folded_average(q, plates)
