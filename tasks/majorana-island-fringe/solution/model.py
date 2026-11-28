from apparatus import Readout
from cavity import cavity_fringe
from island_trace import fringe


class Model(Readout):
    def fringe(self, q):
        if "cavity" in q:
            return cavity_fringe(q, q["parity"])[1]
        return fringe(q, q["parity"])
