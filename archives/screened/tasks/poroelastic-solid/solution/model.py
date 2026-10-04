import numpy as np

POISSON_RATIO = .25
BIOT_COEFFICIENT = .8
BIOT_MODULUS = 2400.
COMPONENTS = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']


class Model:
    def __init__(self):
        self.young_modulus = None

    def fit(self, records):
        experiments = [r['input'] for r in records]
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        self.young_modulus = 0.
        offset = self.predict(experiments)
        self.young_modulus = 1.
        slope = self.predict(experiments)-offset
        self.young_modulus = float(np.sum(slope*(values-offset)/sigma**2)/np.sum((slope/sigma)**2))
        return self

    def predict(self, experiments):
        shear = self.young_modulus/(2*(1+POISSON_RATIO))
        bulk = self.young_modulus/(3*(1-2*POISSON_RATIO))
        sealed_bulk = bulk+BIOT_COEFFICIENT**2*BIOT_MODULUS
        result = []
        for e in experiments:
            strain = np.asarray(e['strain'], dtype=float)
            dilation = np.sum(strain[:3])
            deviator = strain.copy()
            deviator[:3] -= dilation/3
            stress = 2*shear*deviator
            stress[:3] += sealed_bulk*dilation
            result.append(stress[COMPONENTS.index(e['component'])])
        return np.array(result)
