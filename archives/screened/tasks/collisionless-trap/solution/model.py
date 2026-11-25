import numpy as np


def unit_variance(experiments):
    out=[]
    for e in experiments:
        initial=np.array(e['initial_frequencies'])
        final=np.array(e['final_frequencies'])
        angle=e['view_angle']-e['rotation']
        projection=np.array([np.cos(angle),np.sin(angle)])
        out.append(np.sum(projection**2/(initial*final)))
    return np.array(out)


class Model:
    def __init__(self):
        self.temperature=None

    def fit(self, records):
        experiments=[r['input'] for r in records]
        values=np.array([r['value'] for r in records])
        sigma=np.array([r['sigma'] for r in records])
        shape=unit_variance(experiments)
        self.temperature=float(np.clip(np.sum(shape*values/sigma**2)/np.sum(shape**2/sigma**2),.4,1.6))
        return self

    def predict(self, experiments):
        return self.temperature*unit_variance(experiments)
