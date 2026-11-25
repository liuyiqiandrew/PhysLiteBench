"""Gauge-invariant Bloch plaquettes and boundary-energy response."""
from functools import lru_cache
import numpy as np
from scipy.special import spence

TRUE_PARAMETER=1.04


def experiment(mass,handedness=1,temperature=.65):
    return dict(observable='thermal_hall',mass=mass,handedness=handedness,temperature=temperature)


def eigensystem(x,y,mass,handedness):
    x,y=np.broadcast_arrays(x,y);h=np.zeros(x.shape+(2,2),complex)
    d=mass+np.cos(x)+np.cos(y)
    h[...,0,0]=4.5+d;h[...,1,1]=4.5-d
    h[...,0,1]=np.sin(x)-1j*handedness*np.sin(y);h[...,1,0]=h[...,0,1].conj()
    return np.linalg.eigh(h)


@lru_cache(None)
def plaquettes(mass,handedness,order):
    step=2*np.pi/order;k=-np.pi+step*np.arange(order)
    x,y=np.meshgrid(k,k,indexing='ij');_,u=eigensystem(x,y,mass,handedness)
    u10=np.roll(u,-1,axis=0);u01=np.roll(u,-1,axis=1);u11=np.roll(u10,-1,axis=1)
    def link(a,b):return np.sum(a.conj()*b,axis=-2)
    loop=link(u,u10)*link(u10,u11)*link(u11,u01)*link(u01,u)
    curvature=-np.angle(loop)/step**2
    energy,_=eigensystem(x+step/2,y+step/2,mass,handedness)
    return energy,curvature


def edge_current(temperature,energy,curvature):
    x=energy/temperature;z=np.exp(-x)
    # Integrate physical energy carried at each confinement energy offset.
    energy_integral=temperature**2*(spence(1-z)-x*np.log1p(-z))
    return float(np.mean(np.sum(curvature*energy_integral,axis=-1)))


def grid_response(e,scale,order):
    energy,curvature=plaquettes(e['mass'],e['handedness'],order)
    energy=scale*energy;T=e['temperature'];h=.001*T
    return -(edge_current(T-2*h,energy,curvature)-8*edge_current(T-h,energy,curvature)
             +8*edge_current(T+h,energy,curvature)-edge_current(T+2*h,energy,curvature))/(12*h)


@lru_cache(None)
def thermal(mass,handedness,temperature,scale,order=192):
    e=experiment(mass,handedness,temperature)
    return (4*grid_response(e,scale,2*order)-grid_response(e,scale,order))/3


def predict(experiments,scale=TRUE_PARAMETER):
    result=[]
    for e in experiments:
        if e['observable']=='frequency':
            x,y=e['wavevector'];energy,_=eigensystem(x,y,e['mass'],e['handedness']);v=scale*energy[e['band']]
        else:v=thermal(e['mass'],e['handedness'],e['temperature'],scale)
        result.append(v)
    return np.array(result)


def calibration_inputs():
    return [dict(observable='frequency',mass=m,handedness=h,wavevector=k,band=b)
            for _ in range(3) for m in [-1.5,-1.1,-.7] for h in [-1,1]
            for k in [[0.,0.],[.4,1.2],[-1.3,.7],[-2.1,-1.7]] for b in [0,1]]


def hidden_inputs():
    return {'temperature_sweep':[experiment(-1.2,h,T) for h in [-1,1] for T in [.52,.65,.8]],
            'mass_sweep':[experiment(m,h,.65) for m in [-1.55,-1.25,-.85] for h in [-1,1]],
            'orientation_reversal':[experiment(-.7,h,T) for h in [-1,1] for T in [.5,.6,.7]]}
