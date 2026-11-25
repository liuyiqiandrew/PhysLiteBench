import numpy as np
from functools import lru_cache

GAMMA = 1.
ALPHA = .18


@lru_cache(maxsize=4096)
def stationary_orientation(bias, frequency, amplitude):
    detuning = GAMMA*bias-frequency
    drive = GAMMA*amplitude
    damping = ALPHA*frequency
    coefficient = detuning**2+drive**2-damping**2
    discriminant = np.sqrt(coefficient**2+4*damping**2*detuning**2)
    if coefficient >= 0:
        square_z = 2*detuning**2/(coefficient+discriminant)
    else:
        square_z = (-coefficient+discriminant)/(2*damping**2)
    z = np.sqrt(square_z)
    denominator = detuning**2+damping**2*square_z
    x = drive*z*detuning/denominator
    y = -drive*damping*square_z/denominator
    return np.array([x,y,z])


def torque_at(experiment, moment):
    bias = experiment['bias']
    frequency = experiment['frequency']
    amplitude = experiment['amplitude']
    phase = experiment['phase']
    orientation = stationary_orientation(bias,frequency,amplitude)
    c,s = np.cos(phase),np.sin(phase)
    m = np.array([c*orientation[0]-s*orientation[1],
                  s*orientation[0]+c*orientation[1],orientation[2]])
    field = np.array([amplitude*c,amplitude*s,bias])
    return moment*np.cross(m,field)


def predict_at(experiments, moment):
    indices = {'x':0,'y':1,'z':2}
    return np.array([torque_at(e,moment)[indices[e['component']]] for e in experiments])


class Model:
    def __init__(self):
        self.moment = None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        return predict_at(experiments,self.moment)
