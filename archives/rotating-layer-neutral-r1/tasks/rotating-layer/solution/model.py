import numpy as np
from scipy.optimize import minimize_scalar
from scipy.linalg import expm

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
    out=[]
    for e in experiments:
        if e['rotation']==0:
            out.append(height(e,drag_rate));continue
        k=e['wave'];f=e['rotation'];h=e['height']
        initial=np.array([h,-DEPTH*k*e['along_velocity'],k*e['across_velocity']-f*h/DEPTH])
        generator=np.array([[0.,1.,0.],[-(f*f+GRAVITY*DEPTH*k*k),-drag_rate,-DEPTH*f],
                            [-drag_rate*f/DEPTH,0.,-drag_rate]])
        out.append((expm(generator*e['time'])@initial)[0])
    return np.array(out)


class Model:
    def __init__(self):self.drag_rate=None

    def fit(self,records):
        inputs=[r['input'] for r in records]
        values=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        def loss(rate):
            residual=(predict_at(inputs,rate)-values)/sigma
            return residual@residual
        result=minimize_scalar(loss,bounds=(.08,.24),method='bounded',options={'xatol':1e-11})
        self.drag_rate=float(result.x);return self

    def predict(self,experiments):return predict_at(experiments,self.drag_rate)
