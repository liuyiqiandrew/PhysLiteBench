from glass import Readout
from renewal import line

class Model(Readout):
    def memory(self, q):
        return line(q,self.depth_scale)[0]
