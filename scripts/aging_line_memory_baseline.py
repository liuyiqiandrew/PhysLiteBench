from glass import Readout
from clocks import readings


class Model(Readout):
    def memory(self, q):
        # Use the exact single-excursion outcome law but renew the ordinary landscape after each departure.
        return readings(q, self.depth_scale, reset=True)[0]
