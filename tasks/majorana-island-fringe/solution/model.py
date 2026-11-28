from apparatus import Readout
from cavity import cavity_fringe
from island_trace import fringe
from pair_input import pair_fringe


class Model(Readout):
    def fringe(self, q):
        if 'pair_source' in q.get('cavity',{}):
            return pair_fringe(q)[1]
        if "cavity" in q:
            return cavity_fringe(q, q["parity"])[1]
        return fringe(q, q["parity"])
