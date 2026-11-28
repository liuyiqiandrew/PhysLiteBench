from apparatus import Camera
from passive import fringe
from replay import replay


class Model(Camera):
    def fringe(self, q):
        optical = fringe(q, self.mobility)
        if "replay" not in q:
            return optical
        # Preserve the true gate-conditioned spin brightness and optical
        # fringe, but obtain them from independent retained trajectories.
        bright = replay(dict(q, wave=[0.0, 0.0], phase=0.0), self.mobility)
        return optical * bright
