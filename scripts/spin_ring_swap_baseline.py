"""Impose the exact singly monitored marginals but assume independent gap parities in the exchanged ensemble."""
from itertools import product
import numpy as np
from apparatus import Readout
from swap import continuum


class Model(Readout):
    def fringe(self, q):
        if q.get("blank"):
            return np.cos(q["phase"])
        m = len(q["arcs"])
        base = continuum({**q, "records": []})
        means = [continuum({**q, "records": [[[j], 1]]}) / base - 1 for j in range(m)]
        allowed = [
            g
            for g in product((1, -1), repeat=m)
            if all(np.prod([g[i] for i in ix]) == value for ix, value in q["records"])
        ]
        return base * np.mean([np.prod(1 + np.array(g) * means) for g in allowed])
