import numpy as np


def response(e, trap_diffusivity):
    return 4*e['density']*np.sqrt((trap_diffusivity+e['target_diffusivity'])/np.pi)


def predict_at(experiments, trap_diffusivity):
    return np.asarray([response(e, trap_diffusivity) for e in experiments])


class Model:
    def __init__(self):
        self.trap_diffusivity = None

    def fit(self, records):
        raise NotImplementedError('Fit the common trap diffusivity from calibration records.')

    def predict(self, experiments):
        return predict_at(experiments, self.trap_diffusivity)
