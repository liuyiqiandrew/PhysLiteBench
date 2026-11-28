from apparatus import Readout,interferometer


class Model(Readout):
    def fringe(self,q):
        return interferometer(q)[1]
