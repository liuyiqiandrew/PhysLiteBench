"""Independent time-domain normal/superfluid momentum and entropy equations."""
import numpy as np
from scipy.integrate import solve_ivp

PARAMETER='heat_leak'
TRUE_PARAMETER=.16


def experiment(mode=1,time=1.,amplitude=.025):
    return dict(mode=int(mode),time=float(time),amplitude=float(amplitude))


def calibration_inputs():
    return [experiment(0,t,a) for a in [.015,-.025,.03] for t in np.linspace(.1,10.,30)]


def hidden_inputs():
    return {'fundamental':[experiment(1,t) for t in np.linspace(.05,10.,40)],
            'short_wave':[experiment(3,t,-.02) for t in np.linspace(.05,6.,40)],
            'transient':[experiment(k,t,.03) for k in [1,2,3] for t in [.04,.08,.15,.3,.6,1.2]]}


def trajectory(k,times,amplitude,heat_leak):
    def rhs(t,state):
        theta,normal,superfluid=state
        friction=.105*(normal-superfluid)
        # dmu = -s*dT+dp/rho; the periodic pressure gradient is zero
        # for this incompressible zero-total-current longitudinal mode.
        super_acc=-.6*k*theta+friction/.7
        normal_acc=-.7/.3*super_acc
        temperature_acc=-(heat_leak+.03*k*k)*theta/1.5-.6*k*normal/1.5
        return [temperature_acc,normal_acc,super_acc]
    if max(times)==0:return np.repeat(np.array([amplitude,0.,0.])[:,None],len(times),axis=1)
    result=solve_ivp(rhs,(0,max(times)),[amplitude,0.,0.],method='DOP853',t_eval=times,rtol=2e-12,atol=2e-14)
    assert result.success
    return result.y


def predict(experiments,heat_leak):
    result=np.empty(len(experiments));groups={}
    for i,e in enumerate(experiments):groups.setdefault((e['mode'],e['amplitude']),[]).append((i,e))
    for (k,a),rows in groups.items():
        times=sorted(set(e['time'] for _,e in rows));y=trajectory(k,times,a,heat_leak)
        for i,e in rows:result[i]=y[0,times.index(e['time'])]
    return result
