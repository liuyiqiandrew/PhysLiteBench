from apparatus import Readout,interferometer
from partners import partner_map


class Model(Readout):
    """Read the left member of each singlet in the z basis and discard that result."""
    def fringe(self,q):
        return interferometer(q,lambda early,late:partner_map(early,late,resolve=True))[1]
