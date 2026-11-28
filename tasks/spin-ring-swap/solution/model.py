from apparatus import Readout
from swap import continuum


class Model(Readout):
    def fringe(self, q):
        return continuum(q)
