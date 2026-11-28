from magnet import Magnet
from sectors import normalized


class Model(Magnet):
    def measure(self, q):
        # Drop the periodic zero mode from both traces as if the spin insertion could not saturate it.
        return normalized(q, self.gain, drop_zero_mode=True)
