from registry import Printer
from gel import tightened


class Model(Printer):
    @staticmethod
    def dynamics(p, z):
        return tightened(p, z, reset=True)
