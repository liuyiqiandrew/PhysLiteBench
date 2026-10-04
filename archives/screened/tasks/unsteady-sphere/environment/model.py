import numpy as np

RADIUS = .001
FLUID_DENSITY = 1000.
PARTICLE_DENSITY = 2300.
PARTICLE_MASS = PARTICLE_DENSITY*4*np.pi*RADIUS**3/3


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        out = []
        for e in experiments:
            omega = e['frequency']
            fluid_impedance = 6*np.pi*self.viscosity*RADIUS
            velocity = e['force']/(fluid_impedance-1j*omega*PARTICLE_MASS)
            out.append(velocity.real if e['component']=='cosine' else velocity.imag)
        return np.array(out)
