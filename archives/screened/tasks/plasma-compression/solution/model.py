import numpy as np


def prediction(experiments,pressure):
    out=[]
    for e in experiments:
        a=e['radius_ratio'];b=e['length_ratio'];theta=e['theta']
        density=1/(a*a*b)
        magnetic=1/(a*a)
        perpendicular=pressure*density*magnetic
        parallel=pressure*density**3/magnetic**2
        particle=perpendicular*np.sin(theta)**2+parallel*np.cos(theta)**2
        field_stress=magnetic*magnetic*(.5-np.cos(theta)**2)
        out.append(particle+field_stress)
    return np.array(out)


class Model:
    def __init__(self):
        self.pressure=None

    def fit(self,records):
        experiments=[r['input'] for r in records]
        base=prediction(experiments,0.)
        design=prediction(experiments,1.)-base
        values=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        self.pressure=float(np.clip(np.sum(design*(values-base)/sigma**2)/np.sum((design/sigma)**2),.03,.08))
        return self

    def predict(self,experiments):
        return prediction(experiments,self.pressure)
