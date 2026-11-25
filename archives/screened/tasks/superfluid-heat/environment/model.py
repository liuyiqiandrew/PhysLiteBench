import numpy as np
from scipy.optimize import minimize_scalar

DENSITY=1.
NORMAL_DENSITY=.3
SUPERFLUID_DENSITY=.7
TEMPERATURE=1.
SPECIFIC_ENTROPY=.6
HEAT_CAPACITY=1.5
CONDUCTIVITY=.03
MUTUAL_DRAG=.105
COUNTERFLOW_RATE=MUTUAL_DRAG*DENSITY/(NORMAL_DENSITY*SUPERFLUID_DENSITY)
THERMAL_COUPLING=SUPERFLUID_DENSITY/NORMAL_DENSITY*SPECIFIC_ENTROPY**2*TEMPERATURE/HEAT_CAPACITY


def predict_at(experiments,heat_leak):
    effective_conductivity=CONDUCTIVITY+DENSITY*HEAT_CAPACITY*THERMAL_COUPLING/COUNTERFLOW_RATE
    return np.array([e['amplitude']*np.exp(-(heat_leak+effective_conductivity*e['mode']**2)*e['time']/(DENSITY*HEAT_CAPACITY)) for e in experiments])


class Model:
    def __init__(self):self.heat_leak=None

    def fit(self,records):
        raise NotImplementedError

    def predict(self,experiments):return predict_at(experiments,self.heat_leak)
