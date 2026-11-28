from apparatus import Readout
from histories import measurement
from lead import lead_gram


class Model(Readout):
    def fringe(self,q):
        return measurement(q,lead_gram)[1]
