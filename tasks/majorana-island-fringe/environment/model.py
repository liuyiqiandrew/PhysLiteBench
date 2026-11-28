from apparatus import Readout
from island_trace import fringe


class Model(Readout):
    def fringe(self, q):
        # Treat the optical window as independent thinning of the retained shots.
        return fringe(q, q["parity"])
