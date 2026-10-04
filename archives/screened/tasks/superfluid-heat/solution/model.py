import numpy as np
from scipy.linalg import expm
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
    result=[];capacity=DENSITY*HEAT_CAPACITY
    for e in experiments:
        k=e['mode']
        generator=np.array([[-(heat_leak+CONDUCTIVITY*k*k)/capacity,-k/capacity],
                            [capacity*THERMAL_COUPLING*k,-COUNTERFLOW_RATE]])
        result.append(float((expm(generator*e['time'])@np.array([e['amplitude'],0.]))[0]))
    return np.array(result)



class Model:
    def __init__(self):self.heat_leak=None

    def fit(self,records):
        time=np.array([r['input']['time'] for r in records]);amplitude=np.array([r['input']['amplitude'] for r in records])
        value=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        self.heat_leak=float(minimize_scalar(lambda g: np.sum(((amplitude*np.exp(-g*time/(DENSITY*HEAT_CAPACITY))-value)/sigma)**2),bounds=(.08,.3),method='bounded',options={'xatol':1e-13}).x)
        return self

    def predict(self,experiments):return predict_at(experiments,self.heat_leak)
