from current import Calibrated
from geometry import packet_law, observable


class Model(Calibrated):
    def waveform(self, q):
        return observable(q, packet_law(q, self.activity, 12))
