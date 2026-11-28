from meter import Meter
from pulse import measure


class Model(Meter):
    def measure(self,q):
        return measure(q,self.diffusivity)
