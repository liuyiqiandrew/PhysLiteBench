"""Independent direct density-matrix count moment evolution."""
from functools import lru_cache
import numpy as np
from scipy.integrate import solve_ivp

LOWER = np.array([[0.,1.],[0.,0.]])
NUMBER = LOWER.T@LOWER
TRUE_PARAMETER = .73


def count_moments(rabi,detuning,decay,gate,efficiency):
    """Direct density-matrix factorial-moment ODE, no Liouville vectorization."""
    p=rabi*rabi/(decay*decay+2*rabi*rabi+4*detuning*detuning)
    c=1j*rabi*(2*p-1)/(2*(1j*detuning-decay/2))
    rho=np.array([[1-p,c],[c.conjugate(),p]],complex)
    h=np.array([[0.,rabi/2],[rabi/2,detuning]])
    def evolve(_time,state):
        rr=state.reshape((3,2,2));out=[]
        for k in range(3):
            x=rr[k]
            rhs=-1j*(h@x-x@h)+decay*(LOWER@x@LOWER.T-(NUMBER@x+x@NUMBER)/2)
            if k:rhs+=k*efficiency*decay*LOWER@rr[k-1]@LOWER.T
            out.append(rhs)
        return np.array(out).ravel()
    initial=np.zeros((3,2,2),complex);initial[0]=rho
    result=solve_ivp(evolve,[0,gate],initial.ravel(),method='DOP853',rtol=2e-11,atol=2e-13).y[:,-1].reshape(3,2,2)
    mean=float(np.trace(result[1]).real)
    variance=float(np.trace(result[2]).real)+mean-mean**2
    return mean,variance



def experiment(rabi,detuning,decay,duration,observable):
    return dict(rabi=float(rabi),detuning=float(detuning),decay=float(decay),duration=float(duration),observable=observable)


def calibration_inputs():
    settings=[experiment(r,d,g,t,'mean') for r in [.6,1.2,1.8] for d in [-.4,.4]
              for g in [.8,1.2] for t in [1.5,4.5]]
    return settings*6


def hidden_inputs():
    return {
      'mean_anchors':[experiment(*args,'mean') for args in [(.8,0.,1.,2.),(1.5,.3,.9,5.),(1.1,-.2,1.1,3.)]],
      'resonant_gates':[experiment(*args,'variance') for args in [(.6,0.,1.,2.),(1.,0.,1.,4.),(1.8,0.,1.,6.)]],
      'detuned_gates':[experiment(*args,'variance') for args in [(1.,.5,.8,3.),(.8,-.4,1.2,5.),(1.5,-.5,1.,4.)]],
      'duration_scan':[experiment(1.,.2,.9,t,'variance') for t in [1.,3.,6.]],
    }


@lru_cache(maxsize=4096)
def reading(rabi,detuning,decay,duration,observable,efficiency):
    if observable=='mean':
        return efficiency*decay*rabi**2*duration/(decay**2+2*rabi**2+4*detuning**2)
    return count_moments(rabi,detuning,decay,duration,efficiency)[1]


def predict(experiments, efficiency=TRUE_PARAMETER):
    return np.array([reading(e['rabi'],e['detuning'],e['decay'],e['duration'],e['observable'],efficiency) for e in experiments])
