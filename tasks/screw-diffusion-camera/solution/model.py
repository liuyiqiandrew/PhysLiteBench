from apparatus import Camera
from passive import fringe
from replay import replay


class Model(Camera):
    def fringe(self, q):
        if "replay" in q:
            return replay(q, self.mobility)
        return fringe(q, self.mobility)
