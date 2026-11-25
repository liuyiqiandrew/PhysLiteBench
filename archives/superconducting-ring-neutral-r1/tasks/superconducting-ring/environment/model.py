import numpy as np
from scipy.optimize import minimize_scalar


def predict_at(experiments,kinetic_scale):
    radius=np.array([e['radius'] for e in experiments])
    field=np.array([e['magnetic_field'] for e in experiments])
    winding=np.array([e['winding'] for e in experiments])
    log_term=np.log(8*radius/(.012/np.sqrt(radius)))
    geometric=radius*(log_term-2)
    current=(winding-np.pi*radius**2*field)/(geometric+kinetic_scale*radius**2)
    force=current*2*np.pi*radius*field+.5*current**2*(log_term-.5)
    return np.where(np.array([e['observable']=='current' for e in experiments]),current,force)


class Model:
    def __init__(self):self.kinetic_scale=None

    def fit(self,records):
        raise NotImplementedError

    def predict(self,experiments):return predict_at(experiments,self.kinetic_scale)
