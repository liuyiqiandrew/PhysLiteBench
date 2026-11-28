"""An unobserved record distinguishes the occupied site pairs during storage."""
from apparatus import Readout
from echo import reading


class Model(Readout):
    def fringe(self, q):
        return reading(q, distinguish_pairs=True)
