from apparatus import Readout
from lines import measurement
class Model(Readout):
    """Use independent spin outcomes at the two ends of a common singlet."""
    def fringe(self,q):return measurement(q,independent=True)[1]
