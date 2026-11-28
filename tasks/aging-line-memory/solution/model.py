from glass import Readout
from landscape import marked


class Model(Readout):
    def memory(self, q):
        return marked(q, self.depth_scale)[0]
