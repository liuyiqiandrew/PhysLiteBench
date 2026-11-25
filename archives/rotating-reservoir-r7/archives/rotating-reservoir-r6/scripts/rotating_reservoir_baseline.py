"""Equilibrium-averaged collision flux with exact Maxwell heat cumulants."""
from functools import lru_cache
from math import comb
import numpy as np

RATE=3.6
INPUT_FIELDS=('angular_speed','flip_rate','temperature_a','temperature_b','flux_fraction')


def gaussian_moment(degree,mean,temperature):
    result=0.
    for power in range(0,degree+1,2):
        centered=1. if power==0 else np.prod(np.arange(1,power,2))*temperature**(power//2)
        result+=comb(degree,power)*mean**(degree-power)*centered
    return result


def maxwell_statistics(e,drag,rate):
    width=5;generator=np.zeros((2*width,2*width));first=[];second=[]
    for state,sign in enumerate([1.,-1.]):
        u=sign*e['angular_speed'];local=np.zeros((width,width));h1=np.zeros((width,3))
        for degree in range(width):
            local[degree,degree]-=drag*degree+rate
            local[0,degree]+=rate*gaussian_moment(degree,u,e['temperature_b'])
            if degree>=2:local[degree-2,degree]+=drag*e['temperature_a']*degree*(degree-1)
        generator[state*width:(state+1)*width,state*width:(state+1)*width]=local-e['flip_rate']*np.eye(width)
        generator[(1-state)*width:(2-state)*width,state*width:(state+1)*width]=e['flip_rate']*np.eye(width)
        for degree in range(3):
            thermal=0.
            for power in range(degree+1):
                thermal+=comb(degree,power)*u**(degree-power)*gaussian_moment(power+2,0.,e['temperature_b'])
            moment=gaussian_moment(degree,u,e['temperature_b'])
            h1[:3,degree]=rate/2*np.array([u*u*moment-thermal,-2*u*moment,moment])
        polynomial=np.polynomial.Polynomial([-u,1.])
        heat2=rate/4*(polynomial**4-2*e['temperature_b']*polynomial**2+3*e['temperature_b']**2)
        h2=np.zeros(width);h2[:len(heat2.coef)]=heat2.coef
        first.append(h1);second.append(h2)
    forward=generator.T.copy();forward[0]=0.;forward[0,0]=forward[0,width]=1.;rhs=np.zeros(2*width);rhs[0]=1.
    moments=np.linalg.solve(forward,rhs)
    mean=sum(moments[s*width:(s+1)*width]@first[s][:,0] for s in range(2))
    indices=np.r_[np.arange(3),width+np.arange(3)];L=generator[np.ix_(indices,indices)].copy()
    g=np.r_[first[0][:3,0],first[1][:3,0]];constant=np.zeros(6);constant[0]=constant[3]=1.
    target=mean*constant-g;L[0]=moments[indices];target[0]=0.
    phi=np.linalg.solve(L,target).reshape(2,3)
    shot=sum(moments[s*width:(s+1)*width]@second[s] for s in range(2))
    correlation=2*sum(moments[s*width:(s+1)*width]@first[s]@phi[s] for s in range(2))
    return dict(mean=float(mean),noise=float(shot+correlation),shot=float(shot),correlation=float(correlation),
                moments=moments,corrector=phi,stationary_residual=float(max(abs(generator.T@moments))))


def approximate_rate(temperature,flux_fraction):
    return RATE*((1-flux_fraction)+flux_fraction*np.sqrt(4*temperature/np.pi))


def source_statistics(e,drag):
    return maxwell_statistics(e,drag,approximate_rate(e['temperature_b'],e['flux_fraction']))


@lru_cache(4096)
def readout(angular_speed,flip_rate,temperature_a,temperature_b,flux_fraction,statistic,drag):
    e=dict(zip(INPUT_FIELDS,(angular_speed,flip_rate,temperature_a,temperature_b,flux_fraction)))
    result=source_statistics(e,drag)
    if statistic not in ['mean','noise']:raise ValueError('statistic must be mean or noise')
    return result[statistic]


def predict_at(experiments,drag):
    return np.array([readout(*(e[field] for field in INPUT_FIELDS),e.get('statistic','mean'),float(drag)) for e in experiments])


class Model:
    def __init__(self):self.drag=None

    def fit(self,records):
        contrast=np.array([(r['input']['temperature_a']-r['input']['temperature_b'])/r['sigma'] for r in records])
        measured=np.array([r['value']/r['sigma'] for r in records])
        conductance=float(contrast@measured/(contrast@contrast))
        self.drag=float(np.clip(conductance*RATE/(RATE-2*conductance),.4,1.1))
        return self

    def predict(self,experiments):return predict_at(experiments,self.drag)
