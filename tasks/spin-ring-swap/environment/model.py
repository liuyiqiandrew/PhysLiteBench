import numpy as np
from apparatus import Readout


class Model(Readout):
    def fringe(self, q):
        if q.get("blank"):
            return np.cos(q["phase"])
        # In a continuous free-fermion mode subsystem the disjoint-arc purity
        # is the neutral product of chords. This approximation neglects gap records and plates.
        A = np.array(q["arcs"])
        x = A.ravel()
        e = np.tile([1, -1], len(A))
        value = 1.0
        for i, y in enumerate(x):
            for j in range(i):
                if i // 2 != j // 2:
                    value *= (2 * np.sin(np.pi * (y - x[j]))) ** (0.25 * e[i] * e[j])
        return value
