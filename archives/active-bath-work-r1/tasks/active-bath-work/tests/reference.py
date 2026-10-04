"""Finite-mass stationary covariance and measured work, then extrapolation."""
from functools import lru_cache
import numpy as np
from scipy.linalg import expm,solve_continuous_lyapunov

TRUE_PARAMETER = .8
J = np.array([[0.,1.],[-1.,0.]])


@lru_cache(2048)
def finite_state(activity,spring,field,chirality,ratio,mass):
    # Variables (x, sqrt(m)*v, sqrt(m)*f) avoid diverging covariance entries.
    # The drift and diffusion below both have time multiplied by m.
    drift=np.zeros((6,6))
    drift[:2,2:4]=np.sqrt(mass)*np.eye(2)
    drift[2:4,:2]=-spring*np.sqrt(mass)*np.eye(2)
    drift[2:4,2:4]=-np.eye(2)+field*J
    drift[2:4,4:]=np.eye(2)
    drift[4:,4:]=(-np.eye(2)+chirality*J)/ratio
    diffusion=np.zeros((6,6))
    diffusion[2:4,2:4]=1.4*np.eye(2)
    diffusion[4:,4:]=2*activity/ratio**2*np.eye(2)
    covariance=solve_continuous_lyapunov(drift,-diffusion)
    return drift/mass,covariance


def finite_predict(e,activity,mass):
    drift,covariance=finite_state(float(activity),e['spring'],e['field'],e['chirality'],e['memory_ratio'],float(mass))
    if e['readout']=='work':
        # Stationarity: mean integral over duration m equals m*<f dot v>.
        return float(np.trace(covariance[2:4,4:]))
    i,j={'xx':(0,0),'xy':(0,1),'yy':(1,1)}[e['readout']]
    return float((expm(e['lag']*drift)@covariance)[i,j])


def predict(experiments,activity,mass_scale=1.):
    out=[]
    for e in experiments:
        m=mass_scale*.001/max(e['spring'],1.)
        values=[finite_predict(e,activity,m/factor) for factor in (1,2,4)]
        out.append((values[0]-6*values[1]+8*values[2])/3)
    return np.array(out)


def experiment(spring,field,chirality,ratio,readout='work',lag=None):
    e=dict(spring=float(spring),field=float(field),chirality=float(chirality),memory_ratio=float(ratio),readout=readout)
    if lag is not None:e['lag']=float(lag)
    return e


def calibration_inputs():
    controls=[(.8,0.,0.,.2),(1.2,1.,.5,.6),(1.,-1.5,1.2,1.4),(1.5,2.5,-.8,2.8),
              (.6,-.5,-1.8,.4),(1.4,.8,2.5,2.2),(1.1,2.,1.5,.9),(.9,-2.5,-2.,1.7)]
    return [experiment(*control,readout=readout,lag=lag) for _ in range(2)
            for control in controls for lag in [0.,.2,.7,1.8] for readout in ['xx','xy','yy']]


def hidden_inputs():
    return {
        'rapid_force':[experiment(k,b,chi,alpha) for k in [.7,1.4]
                       for b,chi,alpha in [(0.,.8,.2),(.5,-.7,.35),(1.,1.5,.5)]],
        'chiral_alignment':[experiment(k,b,chi,alpha) for k in [.8,1.5]
                            for b,chi,alpha in [(3.,3.,1.),(2.5,2.5,1.),(-3.,-3.,1.)]],
        'opposed_rotations':[experiment(k,b,chi,alpha) for k in [.6,1.3]
                             for b,chi,alpha in [(2.,-2.,1.5),(-1.8,2.4,2.),(2.4,-2.5,2.8)]],
    }
