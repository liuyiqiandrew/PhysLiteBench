"""Independent stochastic impulse propagation and reservoir energy/torque balance."""
from functools import lru_cache
import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad_vec

TRUE_PARAMETER=.67
B_DRAG=.9


def experiment(angular_speed=0.,stiffness_x=1.2,stiffness_y=2.5,temperature_a=1.3,temperature_b=.8):
    return dict(angular_speed=angular_speed,stiffness_x=stiffness_x,stiffness_y=stiffness_y,temperature_a=temperature_a,temperature_b=temperature_b)


def matrices(e,drag):
    w=e['angular_speed'];b=B_DRAG
    drift=np.array([[0.,0.,1.,0.],[0.,0.,0.,1.],[-e['stiffness_x'],-b*w,-drag-b,0.],[b*w,-e['stiffness_y'],0.,-drag-b]])
    noise=np.zeros((4,2));noise[2:,:]=np.sqrt(2*(drag*e['temperature_a']+b*e['temperature_b']))*np.eye(2)
    return drift,noise


@lru_cache(None)
def covariance(angular_speed,stiffness_x,stiffness_y,temperature_a,temperature_b,drag):
    e=experiment(angular_speed,stiffness_x,stiffness_y,temperature_a,temperature_b)
    drift,noise=matrices(e,drag)
    def impulse(t):
        response=expm(drift*t)@noise
        return response@response.T
    value,error=quad_vec(impulse,0,np.inf,epsabs=2e-10,epsrel=2e-10)
    return value


def predict(experiments,drag=TRUE_PARAMETER):
    values=[]
    for e in experiments:
        c=covariance(**e,drag=drag);b=B_DRAG;w=e['angular_speed']
        # Laboratory reservoir energy gain plus its mechanical angular-momentum supply.
        energy_gain=b*(c[2,2]+c[3,3]-w*(c[0,3]-c[1,2]))-2*b*e['temperature_b']
        particle_torque=-b*((c[0,3]-c[1,2])-w*(c[0,0]+c[1,1]))
        values.append(energy_gain+w*particle_torque)
    return np.array(values)


def calibration_inputs():
    return [experiment(0.,kx,ky,a,b) for _ in range(24)
            for kx,ky,a,b in [(.8,2.,1.5,.7),(1.3,2.8,.7,1.3),(1.7,2.2,1.4,.8),(1.1,3.1,.8,1.2),
                             (1.4,2.4,1.6,.6),(.9,2.7,.6,1.4),(1.6,3.,1.3,.8),(1.,2.1,.7,1.1),
                             (1.8,3.2,1.5,.9),(1.2,2.6,.8,1.3),(1.5,2.3,1.4,.7),(.85,2.9,.6,1.2)]]


def hidden_inputs():
    return {'equal_temperatures':[experiment(w,kx,ky,T,T) for w,kx,ky,T in [(.5,1.2,2.5,1.),(-.65,.8,2.,1.2),(.45,1.6,3.1,.8),(-.6,1.4,2.2,1.1)]],
            'temperature_bias':[experiment(w,kx,ky,a,b) for w,kx,ky,a,b in [(-.6,1.1,3.,1.4,.8),(.6,1.6,2.2,.8,1.2),(.65,.9,2.4,1.3,.9),(-.55,1.7,2.8,.7,1.3)]],
            'trap_stiffness':[experiment(w,kx,ky,1.,1.) for w,kx,ky in [(.55,.85,2.1),(-.6,1.8,3.2),(.5,1.1,2.9),(-.65,1.5,2.3)]]}
