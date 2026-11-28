from apparatus import Readout,fringe
from forest import records


class Model(Readout):
    def fringe(self,q):
        return fringe(q,records)[1]
