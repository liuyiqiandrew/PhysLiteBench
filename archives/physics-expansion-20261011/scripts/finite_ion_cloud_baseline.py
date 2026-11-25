import numpy as np


def coefficient(gamma,count):
    if count == 1:
        return max(1-gamma/2,0.)
    free_number = np.clip((2-gamma)/(gamma*(1-1/count)),0.,count)
    return ((1-gamma/2)*free_number-
            gamma*(1-1/count)*free_number**2/4)/count


def predict_at(experiments,coupling):
    return np.asarray([coefficient(e['temperature_multiplier']*coupling,e['ion_count'])
                       for e in experiments])


class Model:
    def __init__(self):
        self.coupling = None

    def fit(self,records):
        design = np.asarray([r['input']['temperature_multiplier']*
                             (r['input']['ion_count']+1)/(4*r['sigma'])
                             for r in records])
        target = np.asarray([(1-r['value'])/r['sigma'] for r in records])
        self.coupling = float(np.clip(design@target/(design@design),.7,1.4))
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.coupling)
