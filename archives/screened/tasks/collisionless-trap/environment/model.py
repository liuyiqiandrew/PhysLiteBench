import numpy as np


def unit_variance(experiments):
    out=[]
    for e in experiments:
        initial=np.array(e['initial_frequencies'])
        final=np.array(e['final_frequencies'])
        angle=e['view_angle']-e['rotation']
        projection=np.array([np.cos(angle),np.sin(angle)])
        temperature_ratio=np.sqrt(np.prod(final)/np.prod(initial))
        out.append(temperature_ratio*np.sum(projection**2/final**2))
    return np.array(out)


class Model:
    def __init__(self):
        self.temperature=None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        return self.temperature*unit_variance(experiments)
