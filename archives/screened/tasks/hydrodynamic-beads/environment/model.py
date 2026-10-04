import numpy as np

BOLTZMANN = 1.380649e-5
TEMPERATURE = 300.
# With lengths in micrometers and forces in pN, eta=0.001 has these units.
VISCOSITY = .001
RADIUS = .5


class Model:
    def __init__(self):
        self.stiffness = None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        out = []
        for e in experiments:
            k = self.stiffness*np.array(e['factors'])
            bead = e['bead']
            if e['measurement'] == 'variance':
                out.append(BOLTZMANN*TEMPERATURE/k[bead])
            else:
                mu = 1/(6*np.pi*VISCOSITY*RADIUS)
                shift = np.array(e['shift'])
                mean = shift*(1-np.exp(-mu*k*e['time']))
                out.append(mean[bead])
        return np.array(out)
