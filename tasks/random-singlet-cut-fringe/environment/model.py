from apparatus import Readout,fringe,all_nested


class Model(Readout):
    def fringe(self,q):
        return fringe(q,all_nested)[1]
