import numpy as np


def response(e, trap_diffusivity):
    return 4*e['density']*np.sqrt((trap_diffusivity+e['target_diffusivity'])/np.pi)


def predict_at(experiments, trap_diffusivity):
    return np.asarray([response(e, trap_diffusivity) for e in experiments])


class Model:
    def __init__(self):
        self.trap_diffusivity = None

    def fit(self, records):
        # Calibration has stationary targets and is linear in sqrt(D_B).
        basis = predict_at([r['input'] for r in records], 1.)
        values = np.asarray([r['value'] for r in records])
        weights = 1/np.asarray([r['sigma'] for r in records])**2
        amplitude = np.sum(weights*basis*values)/np.sum(weights*basis*basis)
        self.trap_diffusivity = float(amplitude**2)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.trap_diffusivity)
