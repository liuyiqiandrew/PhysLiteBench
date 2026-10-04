import numpy as np


def impedance(e,viscosity):
    a=e['radius']
    z=a*np.sqrt(-1j*e['omega']/viscosity)
    return 8*np.pi*viscosity*a**3*(1+z*z/(3*(1+z)))


def spectrum(e,viscosity):
    temperature=e['ambient']+.75*e['rise']
    return float(2*temperature*impedance(e,viscosity).real)


def predict_at(experiments,viscosity):
    return np.asarray([spectrum(e,viscosity) for e in experiments])


class Model:
    def __init__(self):self.viscosity=None

    def fit(self,records):
        raise NotImplementedError('Implement calibration fitting.')

    def predict(self,experiments):return predict_at(experiments,self.viscosity)
