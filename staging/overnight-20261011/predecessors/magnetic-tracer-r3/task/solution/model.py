"""Small-mass charged tracer with a retained linear thermal contact."""
from functools import lru_cache
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.optimize import minimize_scalar

KINDS = {'xx': 0, 'xy': 1, 'yy': 2, 'area': 3}
ROTATION = np.array([[0., 1.], [-1., 0.]])


@lru_cache(4096)
def instantaneous(friction, field, temperature, kx, ky, time, cxx, cxy, cyy):
    mobility = np.array([[friction, field], [-field, friction]])/(friction**2+field**2)
    drift = -mobility@np.diag([kx, ky])
    stationary = np.diag([temperature/kx, temperature/ky])
    delta = np.array([[cxx, cxy], [cxy, cyy]])-stationary
    propagator = expm(time*drift)
    evolved = propagator@delta@propagator.T
    covariance = stationary+evolved
    weight = (ROTATION@drift+(ROTATION@drift).T)/2
    primitive = solve_continuous_lyapunov(drift.T, -weight)
    area = np.trace(primitive@(delta-evolved))
    return np.array([covariance[0,0],covariance[0,1],covariance[1,1],area])


@lru_cache(4096)
def retained(friction, field, temperature, kx, ky, time, cxx, cxy, cyy,
             memory_drag, memory_time, memory_temperature):
    if memory_drag == 0:
        return instantaneous(friction,field,temperature,kx,ky,time,cxx,cxy,cyy)
    mobility = np.array([[friction,field],[-field,friction]])/(friction**2+field**2)
    stiffness = np.diag([kx,ky])
    ratio = memory_drag/memory_time
    drift = np.block([[-mobility@stiffness,mobility],
                      [ratio*mobility@stiffness,-np.eye(2)/memory_time-ratio*mobility]])
    shared = np.vstack([mobility,-ratio*mobility])
    noise = 2*friction*temperature*(shared@shared.T)
    noise[2:,2:] += 2*memory_drag*memory_temperature/memory_time**2*np.eye(2)
    stationary = solve_continuous_lyapunov(drift,-noise)
    initial = np.zeros((4,4))
    initial[:2,:2] = [[cxx,cxy],[cxy,cyy]]
    initial[2:,2:] = memory_drag*memory_temperature/memory_time*np.eye(2)
    delta = initial-stationary
    propagator = expm(time*drift)
    evolved = propagator@delta@propagator.T
    covariance = stationary+evolved
    projector = np.c_[np.eye(2),np.zeros((2,2))]
    raw = projector.T@ROTATION@drift[:2,:]
    weight = (raw+raw.T)/2
    primitive = solve_continuous_lyapunov(drift.T,-weight)
    fast_velocity_area = temperature*np.trace(ROTATION@mobility)
    area = time*(np.trace(weight@stationary)+fast_velocity_area)+np.trace(primitive@(delta-evolved))
    return np.array([covariance[0,0],covariance[0,1],covariance[1,1],area])


def zero_field_calibration(experiments, friction):
    t=np.array([e['time'] for e in experiments]);kx=np.array([e['stiffness'][0] for e in experiments]);ky=np.array([e['stiffness'][1] for e in experiments])
    temp=np.array([e['temperature'] for e in experiments]);c=np.array([e['initial_covariance'] for e in experiments])
    xx=temp/kx+(c[:,0,0]-temp/kx)*np.exp(-2*kx*t/friction)
    xy=c[:,0,1]*np.exp(-(kx+ky)*t/friction)
    yy=temp/ky+(c[:,1,1]-temp/ky)*np.exp(-2*ky*t/friction)
    area=(kx-ky)/(kx+ky)*c[:,0,1]*(1-np.exp(-(kx+ky)*t/friction))
    return np.array([np.array([xx[i],xy[i],yy[i],area[i]])[KINDS[e['readout']]] for i,e in enumerate(experiments)])


def fit_friction(records):
    inputs=[r['input'] for r in records]
    assert all(e['field']==0 and e.get('memory_drag',0)==0 for e in inputs)
    values=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
    def objective(friction):
        residual=(zero_field_calibration(inputs,friction)-values)/sigma
        return residual@residual
    result=minimize_scalar(objective,bounds=(.5,1.5),method='bounded',options={'xatol':1e-12})
    return float(result.x)


def predict_at(experiments,friction):
    if not experiments:return np.empty(0)
    out=[]
    for e in experiments:
        c=e['initial_covariance']
        values=retained(float(friction),float(e['field']),float(e['temperature']),
                        *map(float,e['stiffness']),float(e['time']),float(c[0][0]),float(c[0][1]),float(c[1][1]),
                        float(e.get('memory_drag',0)),float(e.get('memory_time',1)),
                        float(e.get('memory_temperature',e['temperature'])))
        out.append(values[KINDS[e['readout']]])
    return np.array(out)


class Model:
    def __init__(self):self.friction=None
    def fit(self,records):
        self.friction=fit_friction(records)
        return self
    def predict(self,experiments):return predict_at(experiments,self.friction)
