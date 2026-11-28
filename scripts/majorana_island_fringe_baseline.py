from apparatus import Readout
from island_trace import fringe


class Model(Readout):
    """Use the unconstrained Gibbs ensemble for the isolated island."""
    def fringe(self, q):
        return fringe(q, parity=None)
