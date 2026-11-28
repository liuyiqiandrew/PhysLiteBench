from apparatus import Readout
from lines import measurement
class Model(Readout):
    def fringe(self,q):return measurement(q)[1]
