from apparatus import Readout
from resolution import reading
class Model(Readout):
    """The demodulator samples a separate reference with identical detuning statistics."""
    def fringe(self,q):return reading(q,independent_phase=True)[1]
