from apparatus import Readout
from island_trace import fringe


class Model(Readout):
    def fringe(self, q):
        return fringe(q, q["parity"])
