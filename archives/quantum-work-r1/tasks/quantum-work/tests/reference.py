"""Independent Schrödinger propagation and counting-characteristic quadrature."""
from functools import lru_cache
from math import factorial
import numpy as np
from scipy.integrate import solve_ivp

TRUE_PARAMETER=1.06


def experiment(temperature=.6, amplitude_a=.55, amplitude_b=.65, phase=.7,
               time_a=.55, time_b=.45, cumulant=3):
    return dict(temperature=temperature,amplitude_a=amplitude_a,amplitude_b=amplitude_b,
                phase=phase,time_a=time_a,time_b=time_b,cumulant=cumulant)


@lru_cache(maxsize=4096)
def propagate(scale,a,b,phase,t1,t2):
    base=np.diag([0.,scale,2.35*scale]).astype(complex)
    first=base.copy();second=base.copy()
    for i,j,cx,cy in [(0,1,1.,-1j),(0,2,.25,.45j),(1,2,.8,-.7j)]:
        first[i,j]+=a*cx;first[j,i]=first[i,j].conjugate()
        second[i,j]+=b*(np.cos(phase)*cx+np.sin(phase)*cy)
        second[j,i]=second[i,j].conjugate()
    columns=np.eye(3,dtype=complex)
    for h,duration in [(first,t1),(second,t2)]:
        if duration:
            solution=solve_ivp(lambda t,y:(-1j*h@y.reshape(3,3)).ravel(),
                [0.,duration],columns.ravel(),method='DOP853',rtol=2e-12,atol=2e-14)
            if not solution.success:raise RuntimeError(solution.message)
            columns=solution.y[:,-1].reshape(3,3)
    return columns


@lru_cache(maxsize=4096)
def cumulants(temperature,a,b,phase,t1,t2,scale,radius=.08,nodes=32):
    energies=np.array([0.,scale,2.35*scale])
    probability=np.exp(-energies/temperature);probability/=probability.sum()
    rho=np.diag(probability);u=propagate(scale,a,b,phase,t1,t2)
    angles=2*np.pi*np.arange(nodes)/nodes
    values=[]
    for z in radius*np.exp(1j*angles):
        initial=np.diag(np.exp(-z*energies))@rho
        evolved=u@initial@u.conj().T
        values.append(np.trace(np.diag(np.exp(z*energies))@evolved))
    coefficients=np.fft.fft(values)/nodes
    moments=np.array([(factorial(n)*coefficients[n]/radius**n).real for n in [1,2,3]])
    mean,second,third=moments
    return np.array([mean,second-mean**2,third-3*mean*second+2*mean**3])


def predict(experiments,scale=TRUE_PARAMETER):
    return np.array([cumulants(e['temperature'],e['amplitude_a'],e['amplitude_b'],e['phase'],
        e['time_a'],e['time_b'],scale)[e['cumulant']-1] for e in experiments])


def calibration_inputs():
    pulses=[(.4,.55,.6,.3,.4),(.65,.5,-.7,.45,.35),(.5,-.4,1.1,.6,.3),
            (.7,.6,1.5,.35,.55),(-.4,.7,-1.2,.5,.45),(.6,.35,.3,.7,.2)]
    return [experiment(T,a,b,p,t1,t2,n) for _ in range(4)
            for a,b,p,t1,t2 in pulses for T in [.4,.7,1.2] for n in [1,2]]


def hidden_inputs():
    return {
        'temperature_sweep':[experiment(temperature=T) for T in [.4,.65,1.,1.35]],
        'phase_sweep':[experiment(.7,.65,.5,p,.45,.6) for p in [-1.6,-.4,.5,1.4]],
        'duration_sweep':[experiment(.6,.5,.65,.9,t1,t2) for t1,t2 in [(.25,.3),(.4,.45),(.55,.6),(.7,.75)]],
        'lower_cumulants':[experiment(.82,.73,-.43,-1.3,.61,.37,n) for n in [1,2]]
    }
