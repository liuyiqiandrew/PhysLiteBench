from glass import Readout
from renewal import line

class Model(Readout):
    def memory(self, q):
        age, exposure=q['write']
        # The sample keeps its elapsed age, but the write detector is represented
        # by independent thinning. A new conditional population is never selected.
        untagged=dict(q,write=[age+exposure,0.])
        return line(untagged,self.depth_scale)[0]
