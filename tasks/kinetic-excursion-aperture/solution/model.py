from camera import Camera
from excursion import query


class Model(Camera):
    def read(self, q):
        return query(q, self.diffusivity)
