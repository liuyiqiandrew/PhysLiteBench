import numpy as np


def predict_at(experiments,line_density):
    xi = np.asarray([e['bjerrum_length']*e['charge_multiplier']*line_density
                     for e in experiments])
    return 1-xi/2


class Model:
    def __init__(self):
        self.line_density = None

    def fit(self,records):
        design = np.asarray([r['input']['bjerrum_length']*
                             r['input']['charge_multiplier']/2/r['sigma']
                             for r in records])
        target = np.asarray([(1-r['value'])/r['sigma'] for r in records])
        self.line_density = float(np.clip(design@target/(design@design),.7,1.4))
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.line_density)
