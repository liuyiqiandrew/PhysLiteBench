from apparatus import Readout,interferometer
from partners import partner_map


class Model(Readout):
    def fringe(self,q):
        return interferometer(q,partner_map)[1]
