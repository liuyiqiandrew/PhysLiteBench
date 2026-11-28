import numpy as np
from magnet import Magnet
from film import device


def straight(f):
    # On each reversed seam allow one straight domain wall with a uniform transverse position.
    edge = np.sort(np.asarray(f["marks"]), axis=0)
    sx, sy = 1 - 2 * np.sum(edge[1::2] - edge[::2], axis=0)
    traces = np.array([1.0, sy, sx, sx * sy])
    return traces, np.ones(4)


class Model(Magnet):
    def measure(self, q):
        return device(q, self.gain, straight)
