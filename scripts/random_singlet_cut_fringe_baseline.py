from apparatus import Readout,fringe
from forest import records


class Model(Readout):
    """Traverse the right disjoint arc before the left in the carrier circuit."""
    def fringe(self,q):
        return fringe(q,lambda q,a,b:records(q,a,b,reverse_disjoint=True))[1]
