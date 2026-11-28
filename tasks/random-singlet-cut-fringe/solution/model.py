from apparatus import Readout
from renewal import response


class Model(Readout):
    def fringe(self,q):
        return response(q)[1]
