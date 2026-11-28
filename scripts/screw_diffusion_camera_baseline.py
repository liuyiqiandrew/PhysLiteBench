from apparatus import Camera
from angular import fringe


class Model(Camera):
    """Keep both diffusion marginals but give rotation its own independent noise."""
    def fringe(self,q):
        return fringe(q,self.mobility,independent=True)
