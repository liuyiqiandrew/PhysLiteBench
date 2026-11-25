"""Actual bead coordinates, propagated bath impulses and local thermostat heat."""
from functools import lru_cache
import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad_vec

TRUE_PARAMETER=.73
B_DRAG=.9


def experiment(angular_speed=0.,strain_rate=0.,contact_time=.4,stiffness_x=1.2,stiffness_y=2.5,temperature_a=1.3,temperature_b=.9):
    return dict(angular_speed=angular_speed,strain_rate=strain_rate,contact_time=contact_time,stiffness_x=stiffness_x,stiffness_y=stiffness_y,temperature_a=temperature_a,temperature_b=temperature_b)


def matrices(e,drag):
    h=B_DRAG/e['contact_time'];s=e['strain_rate'];w=e['angular_speed']
    drift=np.zeros((6,6));drift[0,2]=drift[1,3]=1
    for i,k in enumerate([e['stiffness_x'],e['stiffness_y']]):
        drift[2+i,i]=-k-h;drift[2+i,2+i]=-drag;drift[2+i,4+i]=h
        drift[4+i,i]=h/B_DRAG;drift[4+i,4+i]=-h/B_DRAG+(s if i==0 else -s)
    drift[4,5]=-w;drift[5,4]=w
    noise=np.zeros((6,4));noise[2,0]=noise[3,1]=np.sqrt(2*drag*e['temperature_a'])
    noise[4,2]=noise[5,3]=np.sqrt(2*e['temperature_b']/B_DRAG)
    return drift,noise


@lru_cache(maxsize=2048)
def covariance(angular_speed,strain_rate,contact_time,stiffness_x,stiffness_y,temperature_a,temperature_b,drag,tolerance=2e-10):
    e=experiment(angular_speed,strain_rate,contact_time,stiffness_x,stiffness_y,temperature_a,temperature_b)
    drift,noise=matrices(e,drag)
    def integrand(t):
        response=expm(drift*t)@noise
        return response@response.T
    C,_=quad_vec(integrand,0,np.inf,epsabs=tolerance,epsrel=tolerance)
    return C


def predict(experiments,drag=TRUE_PARAMETER,tolerance=2e-10):
    result=[]
    extension=np.column_stack([np.eye(2),np.zeros((2,2)),-np.eye(2)])
    for e in experiments:
        C=covariance(**e,drag=drag,tolerance=tolerance)
        h=B_DRAG/e['contact_time']
        result.append(h*h/B_DRAG*np.trace(extension@C@extension.T)-2*h*e['temperature_b']/B_DRAG)
    return np.array(result)


def calibration_inputs():
    records=[]
    for tau in [.2,.4,.6]:
        for kx,ky in [(1.,2.2),(1.6,2.8)]:
            for ta,tb in [(1.4,.8),(.8,1.4)]:
                records.append(experiment(0.,0.,tau,kx,ky,ta,tb))
    for w in [-.5,.5]:
        for tau in [.3,.6]:
            for kx,ky in [(1.1,2.4),(1.5,2.7)]:
                records.append(experiment(w,0.,tau,kx,ky,1.3,.9))
    return records


def hidden_inputs():
    return {
      'deforming_flow':[experiment(w,s,t,kx,ky,1.,1.) for w,s,t,kx,ky in [(.3,.4,.6,1.2,2.5),(-.3,.35,.7,1.4,2.7),(.2,-.35,.6,1.3,2.3),(-.45,-.4,.8,1.5,2.6)]],
      'thermal_bias':[experiment(w,s,t,kx,ky,ta,tb) for w,s,t,kx,ky,ta,tb in [(-.45,-.35,.8,1.5,2.4,1.2,.9),(.2,.3,.7,1.,2.8,.9,1.1),(.4,.4,.6,1.4,2.2,1.3,.8),(-.35,-.4,.7,1.1,2.5,1.,1.2)]],
      'contact_time':[experiment(w,s,t,1.2,2.5,1.1,.9) for w,s,t in [(.25,.4,.4),(-.25,.4,.6),(.25,-.4,.8),(-.5,-.35,.7)]],
      'rotation_anchor':[experiment(w,0.,t,kx,ky,ta,tb) for w,t,kx,ky,ta,tb in [(0.,.3,1.3,2.6,1.2,.8),(.35,.45,1.25,2.45,1.1,.9),(-.55,.7,1.45,2.65,1.3,.85),(0.,.65,1.4,2.3,.85,1.35)]]
    }
