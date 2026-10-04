import numpy as np

RADIUS = .001
FLUID_DENSITY = 1000.
PARTICLE_DENSITY = 2300.
PARTICLE_MASS = PARTICLE_DENSITY*4*np.pi*RADIUS**3/3


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        coefficient = np.array([r['input']['force']/(6*np.pi*RADIUS) for r in records])
        values = np.array([r['value'] for r in records])
        weights = 1/np.array([r['sigma'] for r in records])**2
        self.viscosity = float(np.sum(weights*coefficient**2)/np.sum(weights*coefficient*values))
        return self

    def predict(self, experiments):
        out = []
        for e in experiments:
            omega = e['frequency']
            wave = np.sqrt(-1j*omega*FLUID_DENSITY/self.viscosity)
            fluid_impedance = 6*np.pi*self.viscosity*RADIUS*(1+RADIUS*wave+(RADIUS*wave)**2/9)
            velocity = e['force']/(fluid_impedance-1j*omega*PARTICLE_MASS)
            out.append(velocity.real if e['component']=='cosine' else velocity.imag)
        return np.array(out)
