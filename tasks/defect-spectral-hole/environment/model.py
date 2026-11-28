from apparatus import Readout
from spectrometer import passive


class Model(Readout):
    def fringe(self, q):
        return passive(q)
