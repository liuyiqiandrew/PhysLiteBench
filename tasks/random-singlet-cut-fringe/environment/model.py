from apparatus import Readout,poisson
class Model(Readout):
    def fringe(self,q):return poisson(q)
