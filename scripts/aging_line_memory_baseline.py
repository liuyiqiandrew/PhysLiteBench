from glass import Readout
from landscape import marked


class Model(Readout):
    def memory(self, q):
        # Exact common-sample equilibrium, with the final exposure treated as independent thinning.
        return marked(dict(q, exposure=0.0), self.depth_scale)[0]
