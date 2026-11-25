import numpy as np


def predict_at(experiments,line_density):
    xi = np.asarray([e['bjerrum_length']*e['charge_multiplier']*line_density
                     for e in experiments])
    return 1-xi/2


class Model:
    def __init__(self):
        self.line_density = None

    def fit(self,records):
        raise NotImplementedError('Fit the rod line-charge density.')

    def predict(self,experiments):
        return predict_at(experiments,self.line_density)
