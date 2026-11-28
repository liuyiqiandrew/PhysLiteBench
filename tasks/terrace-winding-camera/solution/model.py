import numpy as np
from camera import image


class Model:
    def fit(self, records):
        self.rate = float(np.mean([row["value"]/(2*row["input"]["height"]) for row in records]))
        return self

    def predict(self, inputs):
        return np.array([2*self.rate*q["height"] if q["mode"]=="wandering" else image(q,self.rate)
                         for q in inputs],dtype=float)
