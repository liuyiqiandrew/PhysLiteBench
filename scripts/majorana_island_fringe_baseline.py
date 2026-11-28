from apparatus import Readout
from counter import measurement


class Model(Readout):
    """The counter has an additional perfect, unobserved record of every tag."""

    def fringe(self,q):
        return measurement(q,record_tags=True)[1]
