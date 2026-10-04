import numpy as np


DEPTH=.05
GRAVITY=9.81


def height(experiment,drag_rate):
    k=experiment['wave'];f=experiment['rotation'];t=experiment['time']
    h=experiment['height'];u=experiment['along_velocity'];v=experiment['across_velocity']
    potential_vorticity=k*v-f*h/DEPTH
    frequency_squared=f*f+GRAVITY*DEPTH*k*k
    frequency=np.sqrt(frequency_squared-drag_rate*drag_rate/4)
    balanced=-DEPTH*f*potential_vorticity/frequency_squared
    derivative=-DEPTH*k*u
    wave=(h-balanced)*np.cos(frequency*t)+(derivative+drag_rate*(h+balanced)/2)*np.sin(frequency*t)/frequency
    return balanced*np.exp(-drag_rate*t)+np.exp(-drag_rate*t/2)*wave


def predict_at(experiments,drag_rate):
    return np.array([height(e,drag_rate) for e in experiments])


class Model:
    def __init__(self):self.drag_rate=None

    def fit(self,records):
        raise NotImplementedError

    def predict(self,experiments):return predict_at(experiments,self.drag_rate)
