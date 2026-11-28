from apparatus import Readout
from dot_core import Island


class Model(Readout):
    """Independent-shot approximation: the final probe starts in thermal equilibrium."""

    def fringe(self,q):
        island=Island(q)
        return island.probe(q['probe'],island.rho,q['phase'])
