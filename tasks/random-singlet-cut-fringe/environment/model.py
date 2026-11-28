from apparatus import Readout,mean_reference
class Model(Readout):
    def fringe(self,q):return mean_reference(q)
