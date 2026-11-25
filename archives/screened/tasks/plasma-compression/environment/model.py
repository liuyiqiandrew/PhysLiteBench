import numpy as np


def prediction(experiments,pressure):
    out=[]
    for e in experiments:
        a=e['radius_ratio'];b=e['length_ratio'];theta=e['theta']
        density=1/(a*a*b)
        magnetic=1/(a*a)
        particle=pressure*density**(5/3)
        field_stress=magnetic*magnetic*(.5-np.cos(theta)**2)
        out.append(particle+field_stress)
    return np.array(out)


class Model:
    def __init__(self):
        self.pressure=None

    def fit(self,records):
        raise NotImplementedError

    def predict(self,experiments):
        return prediction(experiments,self.pressure)
