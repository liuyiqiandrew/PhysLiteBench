from apparatus import Readout
from renewal import response


class Model(Readout):
    """Start a regenerated common bond immediately after both outer removals."""
    def fringe(self,q):
        return response(q,instant=True)[1]
