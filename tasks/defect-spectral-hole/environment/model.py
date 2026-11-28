from apparatus import Readout
from spectrometer import deterministic


class Model(Readout):
    def fringe(self, q):
        return deterministic(q)
