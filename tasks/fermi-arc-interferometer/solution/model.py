from apparatus import Readout
from edge import reading


class Model(Readout):
    def fringe(self, q):
        return reading(q)
