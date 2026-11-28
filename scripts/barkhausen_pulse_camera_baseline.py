from meter import Meter
from pulse import measure


class Model(Meter):
    def measure(self,q):
        result=1.
        for flash in q["flashes"]:
            result*=measure(dict(q,flashes=[flash]),self.diffusivity)
        return result
