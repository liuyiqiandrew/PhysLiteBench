import numpy as np

POISSON_RATIO = .25
BIOT_COEFFICIENT = .8
BIOT_MODULUS = 2400.
COMPONENTS = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']


class Model:
    def __init__(self):
        self.young_modulus = None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        shear = self.young_modulus/(2*(1+POISSON_RATIO))
        lame = self.young_modulus*POISSON_RATIO/((1+POISSON_RATIO)*(1-2*POISSON_RATIO))
        result = []
        for e in experiments:
            strain = np.asarray(e['strain'])
            stress = 2*shear*strain
            stress[:3] += lame*np.sum(strain[:3])
            result.append(stress[COMPONENTS.index(e['component'])])
        return np.array(result)
