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
        coefficient = np.array([BOLTZMANN*TEMPERATURE/r['input']['factors'][r['input']['bead']]
                                for r in records])
        value = np.array([r['value'] for r in records])
        weight = 1/np.array([r['sigma'] for r in records])**2
        self.stiffness = float(np.sum(weight*coefficient**2)/np.sum(weight*coefficient*value))
        return self

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
