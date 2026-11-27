import numpy as np


class Model:
    def fit(self, records):
        raise NotImplementedError

    def predict(self, inputs):
        raise NotImplementedError
