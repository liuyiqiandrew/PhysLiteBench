from magnet import Magnet
from sectors import normalized


class Model(Magnet):
    def measure(self, q):
        return normalized(q, self.gain)
