from glass import Readout, effective_age

class Model(Readout):
    def memory(self, q):
        return effective_age(q,self.depth_scale)
