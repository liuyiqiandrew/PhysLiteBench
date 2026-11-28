from apparatus import Camera
from angular import fringe


class Model(Camera):
    def fringe(self, q):
        # Approximate the centered motor by noise matched to its symmetric diffusion.
        return fringe(q,self.mobility,diffusion_only=True)
