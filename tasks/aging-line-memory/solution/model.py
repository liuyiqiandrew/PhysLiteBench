from glass import Readout
from pump import pump


class Model(Readout):
    def memory(self, q):
        return pump(q, self.depth_scale)[0]
