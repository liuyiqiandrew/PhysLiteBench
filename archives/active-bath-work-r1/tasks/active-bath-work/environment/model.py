import numpy as np
from scipy.linalg import expm

FRICTION = 1.
TEMPERATURE = .7
ROTATION = np.array([[0.,1.],[-1.,0.]])


def predict_at(experiments, activity):
    values = []
    indices = {'xx': (0,0), 'xy': (0,1), 'yy': (1,1)}
    for e in experiments:
        field = e['field']
        if e['readout'] == 'work':
            values.append(2*activity*FRICTION/(e['memory_ratio']*(FRICTION**2+field**2)))
        else:
            mobility = np.linalg.inv(FRICTION*np.eye(2) - field*ROTATION)
            covariance = (TEMPERATURE + activity/(FRICTION*(1+e['chirality']**2))) / e['spring'] * np.eye(2)
            correlation = expm(-e['spring']*mobility*e['lag']) @ covariance
            values.append(correlation[indices[e['readout']]])
    return np.array(values)


class Model:
    def __init__(self):
        self.activity = None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        return predict_at(experiments, self.activity)
