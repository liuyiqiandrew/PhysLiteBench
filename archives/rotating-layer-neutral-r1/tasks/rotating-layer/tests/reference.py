"""Integrate horizontal momentum and layer-volume balance directly."""
from collections import defaultdict
import numpy as np
from scipy.integrate import solve_ivp

TRUE_PARAMETER=.14


def experiment(k,f,h,u,v,t):
    return dict(wave=int(k),rotation=float(f),height=float(h),along_velocity=float(u),across_velocity=float(v),time=float(t))


def calibration_inputs():
    return [experiment(k,0.,h,u,v,t) for k,h,u,v in [(1,.0003,.001,.003),(2,-.0002,-.0015,-.002),(3,.00025,.0005,.004)]
            for t in np.geomspace(.1,35.,48)]


def hidden_inputs():
    times=np.array([.5,1.,2.,4.,8.,12.,20.,30.,40.])
    balanced=[experiment(k,f,h,0.,-9.81*k*h/f,t) for k,f,h in [(1,1.,.0004),(1,2.,.00035),(2,-3.,-.0003)] for t in times]
    mixed=[experiment(k,f,h,u,v,t) for k,f,h,u,v in [(1,1.5,.0003,.001,-.002),(2,-2.,-.0002,.0015,.001),(3,2.5,.0003,-.0015,.002)] for t in times]
    transverse=[experiment(k,f,0.,u,v,t) for k,f,u,v in [(1,2.5,.001,.004),(2,-2.5,-.0015,.005),(1,-1.5,0.,-.004)] for t in times]
    return {'balanced_release':balanced,'mixed_release':mixed,'transverse_release':transverse}


def full_states(experiments,drag_rate,method='DOP853',rtol=2e-12,max_step=np.inf):
    groups=defaultdict(list);out=np.zeros((len(experiments),3))
    for i,e in enumerate(experiments):
        key=tuple(e[k] for k in ['wave','rotation','height','along_velocity','across_velocity'])
        groups[key].append((i,e['time']))
    for (k,f,h,u,v),entries in groups.items():
        times=sorted(set(t for _,t in entries));initial=np.array([h,u,v])
        def derivative(time,y):
            height,along,across=y
            return [-.05*k*along,9.81*k*height+f*across-drag_rate*along,-f*along-drag_rate*across]
        if times[-1]==0:states=initial[:,None]
        else:
            answer=solve_ivp(derivative,(0.,times[-1]),initial,t_eval=times,method=method,rtol=rtol,atol=1e-14,max_step=max_step)
            assert answer.success,answer.message
            states=answer.y
        for index,t in entries:out[index]=states[:,times.index(t)]
    return out


def predict(experiments,drag_rate):return full_states(experiments,drag_rate)[:,0]
