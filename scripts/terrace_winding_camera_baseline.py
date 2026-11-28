"""Use independent retained surfaces at the two exposure heights."""
import numpy as np
from camera import image


def separate(q, rate):
    single_a = dict(q, count_window=[0.,q["circumference"]], count=len(q["bottom"]))
    single_b = dict(q, window=[0.,q["circumference"]])
    a = image(dict(single_a,open_pair=None),rate)
    b = image(dict(single_b,open_pair=None),rate)
    if q.get("open_pair") is None:
        return a*b
    return b*image(single_a,rate)+a*image(single_b,rate)


class Model:
    def fit(self, records):
        self.rate = float(np.mean([row["value"]/(2*row["input"]["height"]) for row in records]))
        return self

    def predict(self, inputs):
        return np.array([2*self.rate*q["height"] if q["mode"]=="wandering" else separate(q,self.rate)
                         for q in inputs],dtype=float)
