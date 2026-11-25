"""Independent time-dependent Hamiltonian evolution and final phase averaging."""
from functools import lru_cache
import numpy as np
from scipy.integrate import solve_ivp

TRUE_TEMPERATURE=.85


def rotation(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s],[s,c]])


@lru_cache(maxsize=128)
def covariance(initial,final,angle,duration=320.):
    initial=np.array(initial);final=np.array(final)
    def rhs(time,flat):
        u=time/duration;s=u*u*(3-2*u)
        frequencies=initial+s*(final-initial)
        r=rotation(s*angle)
        stiffness=(r*frequencies**2)@r.T
        fundamental=flat.reshape(4,4)
        derivative=np.empty_like(fundamental)
        derivative[:2]=fundamental[2:]
        derivative[2:]=-stiffness@fundamental[:2]
        return derivative.ravel()
    answer=solve_ivp(rhs,(0.,duration),np.eye(4).ravel(),method='DOP853',
                     max_step=.15,rtol=3e-11,atol=3e-13)
    assert answer.success
    f=answer.y[:,-1].reshape(4,4)
    initial_covariance=np.diag(np.r_[1/initial**2,[1.,1.]])
    final_covariance=f@initial_covariance@f.T
    r=rotation(angle)
    position=r.T@final_covariance[:2,:2]@r
    momentum=r.T@final_covariance[2:,2:]@r
    energy=(np.diag(momentum)+final**2*np.diag(position))/2
    averaged=(r*(energy/final**2))@r.T
    return averaged,f,energy


def predict(experiments,temperature=TRUE_TEMPERATURE,duration=320.):
    out=[]
    for e in experiments:
        initial=np.array(e['initial_frequencies']);final=np.array(e['final_frequencies'])
        r=rotation(e['rotation']);view=np.array([np.cos(e['view_angle']),np.sin(e['view_angle'])])
        ratio=final/initial
        if abs(ratio[0]-ratio[1])<1e-13:
            # A uniform frequency dilation preserves a common thermal scale.
            stiffness=(r*final**2)@r.T
            c=ratio[0]*np.linalg.inv(stiffness)
        else:
            c=covariance(tuple(initial),tuple(final),e['rotation'],duration)[0]
        out.append(temperature*view@c@view)
    return np.array(out)


def readings(initial,final,rotations,views):
    return [dict(initial_frequencies=list(initial),final_frequencies=list(final),
                 rotation=float(a),view_angle=float(v)) for a in rotations for v in views]


def calibration_inputs():
    return sum([readings([1.,2.5],[s,2.5*s],[-.6,0.,.7],[0.,.4,.9,1.3])
                for s in [.8,.95,1.1,1.35]],[])


def hidden_inputs():
    return {name:readings(initial,final,[-.8,.1,.9],[-1.2,-.6,0.,.4,.9,1.3])
            for name,initial,final in [('split_heating',[1.,2.5],[1.8,2.4]),
                                      ('compression_and_expansion',[.8,3.2],[1.5,2.]),
                                      ('opposing_mode_work',[1.5,2.2],[.7,3.4])]}
