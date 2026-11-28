from glass import Readout
from clocks import readings


class Model(Readout):
    def memory(self, q):
        return readings(q, self.depth_scale)[0]
