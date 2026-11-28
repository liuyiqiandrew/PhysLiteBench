import numpy as np

from apparatus import Camera, ROT, SCREWS, TRANS


class Model(Camera):
    def fringe(self, q):
        # Isotropic Gaussian displacement with the correct short-time variance.
        screw = q["hand"]*SCREWS + np.cross(np.eye(3), q["tag"])
        diffusivity = self.mobility * (TRANS.sum() + ROT @ (screw**2).sum(axis=1))/3
        return np.cos(q["phase"]) * np.exp(-diffusivity * np.dot(np.square(q["wave"]), q["times"]))
