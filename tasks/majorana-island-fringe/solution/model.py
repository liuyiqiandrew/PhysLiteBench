from apparatus import Readout
from cavity import cavity_fringe
from island_trace import fringe
from pair_input import pair_fringe
from dispersive import dispersive
from dots import dot_fringe


class Model(Readout):
    def fringe(self, q):
        if 'dots' in q:
            return dot_fringe(q)[1]
        if 'dispersive' in q.get('cavity',{}):
            return dispersive(q)[1]
        if 'pair_source' in q.get('cavity',{}):
            return pair_fringe(q)[1]
        if "cavity" in q:
            return cavity_fringe(q, q["parity"])[1]
        return fringe(q, q["parity"])
