from apparatus import Readout
from counter import measurement


class Model(Readout):
    def fringe(self,q):
        return measurement(q)[1]
