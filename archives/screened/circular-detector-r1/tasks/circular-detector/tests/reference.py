"""Independent cylindrical vacuum-mode sum for circular detector response."""
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.special import jv

TRUE_PARAMETER=.012


def experiment(trajectory,a,gap,readout='excitation',speed=None):
    result=dict(trajectory=trajectory,acceleration=float(a),gap=float(gap),readout=readout)
    if trajectory=='circle':result['speed']=float(speed)
    return result


def calibration_inputs():
    return [experiment('cusp',a,a*ratio,readout) for _ in range(3) for a in [.7,1.,1.4,1.7] for ratio in [.15,.3,.5,.8,1.1,1.4] for readout in ['excitation','deexcitation']]


def hidden_inputs():
    return {
        'speed_variation':[experiment('circle',1.,ratio,speed=v) for v in [.3,.36,.42] for ratio in [.7,.9,1.1]],
        'acceleration_variation':[experiment('circle',a,a*ratio,speed=.35) for a in [.8,1.2,1.6] for ratio in [.65,.9,1.15]],
        'small_gaps':[experiment('circle',1.1,1.1*ratio,speed=v) for v in [.3,.34,.38] for ratio in [.12,.18,.24]]}


@lru_cache(maxsize=1024)
def mode_rate(acceleration,speed,signed_gap,maximum_mode=400,order=80):
    gamma=1/np.sqrt(1-speed**2)
    radius=gamma**2*speed**2/acceleration
    angular_frequency=speed/radius
    first=int(np.floor(signed_gap/(gamma*angular_frequency)))+1
    mode=np.arange(first,maximum_mode+1)
    frequency=mode*angular_frequency-signed_gap/gamma
    nodes,weights=leggauss(order)
    theta=(nodes+1)*np.pi/4
    weights=weights*np.pi/4
    density=jv(mode[:,None],frequency[:,None]*radius*np.sin(theta))**2
    angular_integral=np.sum(density*weights*np.sin(theta),axis=1)
    return float(frequency@angular_integral/(2*np.pi*gamma))


@lru_cache(maxsize=128)
def cusp_rate(acceleration,signed_gap):
    coefficient=acceleration**2/12
    regular=quad(lambda s:coefficient/(1+coefficient*s*s),0,np.inf,weight='cos',wvar=abs(signed_gap),epsabs=2e-13,limlst=200)[0]/(2*np.pi**2)
    return regular+max(-signed_gap,0)/(2*np.pi)


def predict(experiments,coupling=TRUE_PARAMETER):
    result=[]
    for e in experiments:
        gap=e['gap']*(1 if e['readout']=='excitation' else -1)
        value=cusp_rate(e['acceleration'],gap) if e['trajectory']=='cusp' else mode_rate(e['acceleration'],e['speed'],gap)
        result.append(coupling*value)
    return np.array(result)
