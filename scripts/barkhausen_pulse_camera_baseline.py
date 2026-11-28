from meter import Meter, windows
import numpy as np
from pulse import measure


class Model(Meter):
    def measure(self,q):
        # Observe the second gate on a fresh sample with both original stop sizes.
        x,side,_,_=q["gates"][0]
        new=[x,side,0.,10.]
        return measure(dict(q,gates=[new,q["gates"][1]]),self.diffusivity)
