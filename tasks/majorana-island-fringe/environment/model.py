from apparatus import Readout
from histories import measurement


class Model(Readout):
    """Calibrated model for a decoupled lead."""
    def fringe(self,q):
        return measurement(q)[1]
