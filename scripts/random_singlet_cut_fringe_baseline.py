from apparatus import Readout
from renewal import response


class Model(Readout):
    """Give the herald copy its own independently drawn chain."""
    def fringe(self,q):
        return response(q,independent_herald=True)[1]
