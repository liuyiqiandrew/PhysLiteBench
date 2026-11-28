from glass import Readout
from pump import pump


class Model(Readout):
    def memory(self, q):
        # Retain the shared specimen and selected line, but leave its lifetime unchanged after illumination.
        return pump(q, self.depth_scale, control="passive")[0]
