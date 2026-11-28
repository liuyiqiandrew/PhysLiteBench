import numpy as np


ROT = np.array([.55, .82, 1.12])
TRANS = np.array([.18, .27, .36])
SCREWS = np.array([[.27,.46,-.18], [.04,-.23,.33], [-.38,.17,.29]])


def rotation(q, mobility):
    return np.exp(-mobility * (ROT.sum()-ROT[q["axis"]]) * q["t"])


class Camera:
    def fit(self, records):
        slopes = [(ROT.sum()-ROT[r["input"]["axis"]])*r["input"]["t"] for r in records]
        measured = np.log([r["value"] for r in records])
        self.mobility = float(np.clip(-np.dot(slopes, measured)/np.dot(slopes, slopes), .7, 1.2))
        return self

    def predict(self, inputs):
        return np.array([rotation(q,self.mobility) if q["mode"]=="rotation"
                         else self.fringe(q) for q in inputs], dtype=float)
