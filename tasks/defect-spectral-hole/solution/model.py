from apparatus import Readout
from echo import reading


class Model(Readout):
    def fringe(self, q):
        return reading(q)
