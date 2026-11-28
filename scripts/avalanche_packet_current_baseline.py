"""Use the exact marginal packet heights at both contacts but treat them as independent."""
from current import Calibrated
from geometry import packet_law, observable


class Model(Calibrated):
    def waveform(self, q):
        return observable(q, packet_law(q, self.activity, 12), independent=True)
