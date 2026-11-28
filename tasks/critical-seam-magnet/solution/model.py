from magnet import Magnet
from sectors import blocks
from film import device


class Model(Magnet):
    def measure(self, q):
        return device(q, self.gain, blocks)
