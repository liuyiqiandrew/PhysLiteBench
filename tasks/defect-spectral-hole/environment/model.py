from apparatus import Readout, gaussian


class Model(Readout):
    def fringe(self, q):
        return gaussian(q)
