"""Resolve the acoustic records but attribute every detected flip to a distinct defect."""
from apparatus import Readout
from echo import reading


class Model(Readout):
    def fringe(self, q):
        return reading(q, independent_emitters=True)
