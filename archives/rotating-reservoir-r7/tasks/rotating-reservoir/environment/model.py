"""Completed rotor mean/noise model with a Gaussian current readout."""
from functools import lru_cache
from math import comb
import numpy as np
from scipy.linalg import solve

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


def bernoulli(value):
    result=np.empty_like(value);small=abs(value)<1e-5
    result[small]=1-value[small]/2+value[small]**2/12-value[small]**4/720
    result[~small]=value[~small]/np.expm1(value[~small])
    return result


@lru_cache(32)
def collision_grid(temperature_a,temperature_b,angular_speed,flux_fraction,points,extent):
    width=extent*np.sqrt(max(temperature_a,temperature_b))+abs(angular_speed)
    edges=np.linspace(-width,width,points+1);x=(edges[1:]+edges[:-1])/2;spacing=edges[1]-edges[0]
    v=x[:,None];g=x[None,:];jumps=[];first=[];second=[];motor=[]
    for sign in [1.,-1.]:
        flow=sign*angular_speed
        density=np.exp(-(x-flow)**2/(2*temperature_b))/np.sqrt(2*np.pi*temperature_b)*spacing
        jump=RATE*((1-flux_fraction)+flux_fraction*abs(v-g))*density[None,:]
        heat=.5*((v-flow)**2-(g-flow)**2)
        jumps.append(jump);first.append(jump*heat);second.append(np.sum(jump*heat**2,axis=1))
        motor.append(np.sum(jump*flow*(g-v),axis=1))
    return x,edges,spacing,jumps,first,second,motor


def grid_statistics(e,drag,points=241,extent=9.,details=False):
    x,edges,h,jumps,first,second,motor=collision_grid(e['temperature_a'],e['temperature_b'],e['angular_speed'],e['flux_fraction'],points,extent)
    if drag==0.:
        continuous=np.zeros((points,points))
    else:
        diffusion=drag*e['temperature_a'];drift=-drag*edges[1:-1]
        up=diffusion/h**2*bernoulli(-drift*h/diffusion)
        down=diffusion/h**2*bernoulli(drift*h/diffusion)
        continuous=np.diag(up,1)+np.diag(down,-1)
        continuous-=np.diag(continuous.sum(axis=1))
    blocks=[continuous+jump-np.diag(jump.sum(axis=1)) for jump in jumps]
    identity=np.eye(points);flip=e['flip_rate']
    generator=np.block([[blocks[0]-flip*identity,flip*identity],[flip*identity,blocks[1]-flip*identity]])
    forward=generator.T.copy();forward[0]=1.;rhs=np.zeros(2*points);rhs[0]=1.
    probability=solve(forward,rhs,check_finite=False)
    g=np.r_[first[0].sum(axis=1),first[1].sum(axis=1)];b=np.r_[second[0],second[1]]
    mean=float(probability@g);system=generator.copy();pivot=int(np.argmax(probability));system[pivot]=probability
    rhs=mean-g;rhs[pivot]=0.;phi=solve(system,rhs,check_finite=False)
    correlation=np.r_[first[0]@phi[:points],first[1]@phi[points:]]
    noise=float(probability@(b+2*correlation))
    result=dict(mean=mean,noise=noise)
    if details:
        result.update(shot=float(probability@b),correlation=float(2*probability@correlation),probability=probability,grid=x,
            minimum_probability=float(probability.min()),edge_probability=float(probability[[0,points-1,points,2*points-1]].sum()),
            outer_two_sigma_probability=float(probability[np.tile(abs(x)>abs(e['angular_speed'])+7*np.sqrt(max(e['temperature_a'],e['temperature_b'])),2)].sum()),
            normalization_error=float(abs(probability.sum()-1)),conditional_mass_error=float(abs(probability[:points].sum()-.5)),
            stationary_residual=float(max(abs(generator.T@probability))),poisson_residual=float(max(abs(generator@phi-(mean-g)))),
            motor=float(probability@np.r_[motor[0],motor[1]]),velocity_square=float(probability@np.tile(x*x,2)),
            bath_a_discrete=float(-probability@np.tile(continuous@(.5*x*x),2)))
    return result


def physical_statistics(e,drag,points=241,extent=9.):
    if e['flux_fraction']==0.:
        return maxwell_statistics(e,drag,RATE)
    coarse=grid_statistics(e,drag,points,extent)
    fine=grid_statistics(e,drag,2*points,extent)
    return {key:float((4*fine[key]-coarse[key])/3) for key in ['mean','noise']}


def scgf(e,drag,heat_bias):
    values=physical_statistics(e,drag)
    return float(-heat_bias*values['mean']+heat_bias**2*values['noise']/2)


@lru_cache(4096)
def readout(angular_speed,flip_rate,temperature_a,temperature_b,flux_fraction,statistic,drag,heat_bias=0.):
    e=dict(zip(INPUT_FIELDS,(angular_speed,flip_rate,temperature_a,temperature_b,flux_fraction)))
    if statistic=='scgf':
        return scgf(e,drag,float(heat_bias))
    if statistic not in ['mean','noise']:raise ValueError('statistic must be mean, noise or scgf')
    return physical_statistics(e,drag)[statistic]


def predict_at(experiments,drag):
    return np.array([readout(*(e[field] for field in INPUT_FIELDS),e.get('statistic','mean'),float(drag),float(e.get('heat_bias',0.))) for e in experiments])


class Model:
    def __init__(self):self.drag=None

    def fit(self,records):
        contrast=np.array([(r['input']['temperature_a']-r['input']['temperature_b'])/r['sigma'] for r in records])
        measured=np.array([r['value']/r['sigma'] for r in records])
        conductance=float(contrast@measured/(contrast@contrast))
        self.drag=float(np.clip(conductance*RATE/(RATE-2*conductance),.4,1.1))
        return self

    def predict(self,experiments):return predict_at(experiments,self.drag)
