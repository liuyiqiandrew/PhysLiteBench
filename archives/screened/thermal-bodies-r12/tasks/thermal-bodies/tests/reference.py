"""Direct outer-thermostat heat moments in canonical position/momentum order."""
from functools import lru_cache
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.integrate import solve_ivp

TRUE_PARAMETER=.7


@lru_cache(512)
def state(gamma,k1,k2,c,tau,mass,t1,t2):
    main=np.array([[k1+c,-c],[-c,k2+c]])
    link=gamma/tau
    potential=np.block([[main+link*np.eye(2),-link*np.eye(2)],[-link*np.eye(2),link*np.eye(2)]])
    inverse_mass=np.diag([1.,1.,1/mass,1/mass])
    drag=np.diag([0.,0.,gamma,gamma])
    drift=np.block([[np.zeros((4,4)),inverse_mass],[-potential,-drag@inverse_mass]])
    noise=np.zeros((8,8));noise[6,6]=2*gamma*t1;noise[7,7]=2*gamma*t2
    covariance=solve_continuous_lyapunov(drift,-noise)
    return drift,noise,covariance,np.linalg.eigh(main)[1]


def heat_moments(e,gamma,rtol=2e-11):
    drift,noise,S,_=state(float(gamma),*e['springs'],e['coupling'],e['memory'],e['coupler_mass'],*e['temperatures'])
    i=e['bath'];index=6+i;T=e['temperatures'][i];mass=e['coupler_mass'];duration=e['duration']
    a=gamma*T/mass;b=gamma/mass**2
    column=S[:,index];unit=np.eye(8)[:,index]
    # Ito q drift and the correlation of its heat noise with the same bath's force.
    source=a*S-b*(S[index,index]*S+2*np.outer(column,column))
    source+=2*gamma*T/mass*(np.outer(unit,column)+np.outer(column,unit))
    current=a-b*S[index,index]
    def ode(time,z):
        mean=z[0];mixed=z[1:65].reshape(8,8)
        derivative=drift@mixed+mixed@drift.T+noise*mean+source
        second=2*a*mean-2*b*mixed[index,index]+2*gamma*T/mass**2*S[index,index]
        return np.r_[current,derivative.ravel(),second]
    if duration==0:return 0.,0.
    answer=solve_ivp(ode,(0.,duration),np.zeros(66),method='DOP853',rtol=rtol,atol=rtol/100)
    assert answer.success,answer.message
    mean=answer.y[0,-1];variance=answer.y[-1,-1]-mean**2
    return float(mean),float(variance)


def predict(experiments,gamma):
    out=[]
    for e in experiments:
        A,D,S,vectors=state(float(gamma),*e['springs'],e['coupling'],e['memory'],e['coupler_mass'],*e['temperatures'])
        if e['readout']=='heat_variance':value=heat_moments(e,gamma)[1]
        elif e['readout']=='heat_current':
            i=e['bath'];M=e['coupler_mass'];value=gamma*(e['temperatures'][i]/M-S[6+i,6+i]/M**2)
        else:
            v=vectors[:,e['mode']];value=v@(expm(e['lag']*A)@S)[4:6,4:6]@v
        out.append(value)
    return np.array(out)


def preparation(springs,coupling,memory,mass,temperatures):
    return dict(springs=springs,coupling=coupling,memory=memory,coupler_mass=mass,temperatures=temperatures)


def calibration_inputs():
    controls=[preparation([1.,1.2],.2,.3,.08,[1.5,.6]),preparation([.8,1.4],.5,.8,.16,[.6,1.8]),
              preparation([1.3,.9],.9,1.7,.25,[2.,1.]),preparation([1.1,1.1],.35,1.2,.12,[1.2,1.2])]
    inputs=[]
    for _ in range(2):
        for e in controls:
            for mode in [0,1]:
                for lag in [.15,.35,.6,.9,1.2,1.6,2.,2.5,3.,3.7,4.5,5.5]:
                    inputs.append(dict(e,readout='mode_correlation',mode=mode,lag=lag))
    for _ in range(4):
        for e in controls:
            for bath in [0,1]:inputs.append(dict(e,readout='heat_current',bath=bath))
    return inputs


def hidden_inputs():
    controls=[('equilibrium_windows',preparation([1.,1.2],.4,.3,.08,[1.2,1.2]),[.05,.12,.3,.65]),
              ('unequal_temperatures',preparation([.8,1.4],.6,.7,.16,[1.8,.6]),[.1,.3,.7,1.5]),
              ('long_coupling_time',preparation([1.3,.9],.15,1.7,.25,[.7,1.6]),[.2,.6,1.2,2.5])]
    return {name:[dict(e,readout='heat_variance',bath=i,duration=time) for i in [0,1] for time in times]
            for name,e,times in controls}
