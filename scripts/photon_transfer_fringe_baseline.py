"""Exact single-source squeezed photon statistics followed by independent ray routing."""
import numpy as np

from optics import IndependentRays


class Model(IndependentRays):
    def source_pgf(self, source, argument):
        n, r = source[:, 0], source[:, 1]
        mean = n + (2*n+1)*np.sinh(r)**2
        pairing = (n+.5)*np.sinh(2*r)
        plus = 1 + (mean+pairing)*(1-argument)
        minus = 1 + (mean-pairing)*(1-argument)
        return 1 / np.prod(np.sqrt(plus)*np.sqrt(minus))
