from single import independent
from glass import Readout


class Model(Readout):
    def memory(self, q):
        return independent(q, self.depth_scale)
