from functools import lru_cache
import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from scipy.optimize import minimize_scalar

B_DRAG=.9


@lru_cache(maxsize=512)
def stationary_covariance(angular_speed,strain_rate,contact_time,stiffness_x,stiffness_y,temperature_a,temperature_b,drag):
    flow=np.array([[strain_rate,-angular_speed],[angular_speed,-strain_rate]])
    drift=np.zeros((6,6))
    drift[:2,2:4]=np.eye(2)
    drift[2:4,:2]=-np.diag([stiffness_x,stiffness_y])
    drift[2:4,2:4]=-drag*np.eye(2)
    drift[2:4,4:]=np.eye(2)
    drift[4:,:2]=B_DRAG*flow/contact_time
    drift[4:,2:4]=-B_DRAG*np.eye(2)/contact_time
    drift[4:,4:]=flow-np.eye(2)/contact_time
    noise=np.diag([0.,0.,2*drag*temperature_a,2*drag*temperature_a,
                   2*B_DRAG*temperature_b/contact_time**2,2*B_DRAG*temperature_b/contact_time**2])
    return solve_continuous_lyapunov(drift,-noise)


def heat_rate(experiment,drag):
    covariance=stationary_covariance(**experiment,drag=drag)
    return np.trace(covariance[4:,4:])/B_DRAG-2*experiment['temperature_b']/experiment['contact_time']


def predict_at(experiments,drag):
    return np.array([heat_rate(e,drag) for e in experiments],dtype=float)


class Model:
    def __init__(self):
        self.drag=None

    def fit(self,records):
        experiments=[r['input'] for r in records]
        observed=np.array([r['value'] for r in records])
        sigma=np.array([r['sigma'] for r in records])
        def objective(drag):
            residual=(predict_at(experiments,drag)-observed)/sigma
            return float(residual@residual)
        result=minimize_scalar(objective,bounds=(.4,1.1),method='bounded',options={'xatol':1e-12})
        self.drag=min([.4,float(result.x),1.1],key=objective)
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.drag)
