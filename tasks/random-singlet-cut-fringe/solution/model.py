from apparatus import Readout
from resolution import reading
class Model(Readout):
    def fringe(self,q):return reading(q)[1]
