from glass import Readout, one_temperature


class Model(Readout):
    def memory(self, q):
        return one_temperature(q, self.depth_scale)
