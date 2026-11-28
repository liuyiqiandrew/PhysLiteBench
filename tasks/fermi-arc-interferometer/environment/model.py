import numpy as np
from apparatus import Readout


class Model(Readout):
    def fringe(self, q):
        if q.get("blank"):
            return np.cos(q["phase"])
        arcs = np.asarray(q["arcs"])
        lengths = arcs[:, 1] - arcs[:, 0]
        offsets = arcs[:, 3] - arcs[:, 2]
        return np.prod(np.cos(np.pi * (q["detuning"] * lengths + offsets)))
