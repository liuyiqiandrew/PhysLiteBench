"""Two-radius radial oscillator heat kernel with angular Fourier order."""
from functools import lru_cache
import numpy as np
from scipy.special import iv, roots_genlaguerre

TRUE_PARAMETER = 1.09


def experiment(k=1., temperature=1., probe=1., radial=0.):
    return dict(stiffness=k,temperature=temperature,probe=probe,radial_probe=radial)


def calibration_inputs():
    angular = [experiment(k,t,p,0.) for k in [.8,1.2,1.6]
               for t in [.7,1.3] for p in [-1.5,-.5,.5,1.5]]
    radial = [experiment(k,t,0.,s) for k in [.8,1.2,1.6]
              for t in [.7,1.3] for s in [.3,.9,1.5]]
    return angular+radial


def hidden_inputs():
    return {
        'shared_anchors': [experiment(k,t,p,s) for k,t in [(.7,.8),(1.1,1.6),(1.7,1.1)]
                           for p,s in [(-2.,0.),(.3,0.),(0.,1.2),(0.,0.)]],
        'joint_positive': [experiment(k,t,p,s) for k,t,s in [(.7,1.5,1.2),(1.,1.,.8),(1.5,.7,1.6)]
                           for p in [.6,1.2,2.]],
        'joint_negative': [experiment(k,t,p,s) for k,t,s in [(.8,1.6,.7),(1.3,1.2,1.4),(1.7,.8,1.8)]
                           for p in [-.8,-1.4,-2.3]],
        'joint_strong_radial': [experiment(k,t,p,s) for k,t,s in [(.6,1.8,1.8),(.9,1.4,1.7),(1.2,1.7,1.5)]
                                for p in [.4,1.,1.8]],
    }


@lru_cache(maxsize=4096)
def log_characteristic(probe, radial, mu, k, temperature, time, order=64):
    a=mu*k; D=mu*temperature;nu=abs(probe)
    omega=np.sqrt(a*a+4*D*radial)
    tau=omega*time
    coefficient=(a+omega/np.tanh(tau))/(4*D)
    z=omega/(2*D*np.sinh(tau)*coefficient)
    x,w=roots_genlaguerre(order,nu/2)
    p=z*np.sqrt(x[:,None]*x[None,:])/2
    normalized=iv(nu,2*p)/p**nu
    integral=float(w@normalized@w)
    return a*time+np.log(a*omega/(8*D*D*coefficient**2))-np.log(np.sinh(tau))+nu*np.log(z/2)+np.log(integral)


def predict(experiments, mobility, times=(10.,14.), order=64):
    values = []
    for e in experiments:
        k,temperature = e['stiffness'],e['temperature']
        probe,radial = e['probe'],e['radial_probe']
        if probe == 0 and radial == 0:
            values.append(0.)
            continue
        omega = np.sqrt((mobility*k)**2+4*mobility*temperature*radial)
        t1,t2 = np.asarray(times)/omega
        log1 = log_characteristic(probe,radial,mobility,k,temperature,t1,order)
        log2 = log_characteristic(probe,radial,mobility,k,temperature,t2,order)
        values.append(-(log2-log1)/(t2-t1))
    return np.asarray(values)
