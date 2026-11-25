"""Independent Chebyshev collocation of the continuous rotor jump generator."""
import numpy as np
from scipy.linalg import solve
from numpy.polynomial.legendre import leggauss
from functools import lru_cache

def interpolation(nodes,weights,targets):
 delta=targets[:,None]-nodes[None,:]
 near=abs(delta)<1e-13
 safe=np.where(near,1.,delta)
 value=weights[None,:]/safe
 value/=value.sum(axis=1)[:,None]
 for i in np.flatnonzero(near.any(axis=1)):
  value[i]=0;value[i,np.argmax(near[i])]=1
 return value

@lru_cache(2048)
def state(gamma=.67,ta=1.,tb=1.,omega=.8,flip=1.,eta=1.,points=121,extent=9.,quadrature=140):
 width=extent*np.sqrt(max(ta,tb))+abs(omega)
 x=width*np.cos(np.arange(points)*np.pi/(points-1))
 w=(-1.)**np.arange(points);w[[0,-1]]*=.5
 delta=x[:,None]-x[None,:]
 D=(w[None,:]/w[:,None])/np.where(np.eye(points,dtype=bool),1.,delta)
 np.fill_diagonal(D,0);np.fill_diagonal(D,-D.sum(axis=1))
 # Enforce reflecting backward-Neumann boundaries by eliminating endpoint
 # values. Interior test functions remain freely parameterized.
 E=np.zeros((points,points-2));E[1:-1]=np.eye(points-2)
 E[[0,-1]]=solve(D[np.ix_([0,points-1],[0,points-1])],-D[[0,-1],1:-1])
 L=(-gamma*x[:,None]*D+gamma*ta*(D@D))@E
 nodes,weights=leggauss(quadrature);blocks=[];marks=[];squared=[];motors=[];size=points-2
 for sign in [1.,-1.]:
  u=sign*omega;J=np.zeros((size,size));A=np.zeros_like(J);b=np.zeros(size);motor=np.zeros(size)
  for i,v in enumerate(x[1:-1]):
   for lo,hi in [(-width,v),(v,width)]:
    g=(lo+hi)/2+(hi-lo)*nodes/2
    phi=np.exp(-(g-u)**2/(2*tb))/np.sqrt(2*np.pi*tb)
    rates=3.6*((1-eta)+eta*abs(v-g))*phi*weights*(hi-lo)/2
    mat=interpolation(x,w,g)@E
    q=.5*((v-u)**2-(g-u)**2)
    J[i]+=rates@mat;J[i,i]-=rates.sum()
    A[i]+=(rates*q)@mat;b[i]+=rates@q**2;motor[i]+=rates@(u*(g-v))
  blocks.append(L[1:-1]+J);marks.append(A);squared.append(b);motors.append(motor)
 I=np.eye(size)
 B=np.block([[blocks[0]-flip*I,flip*I],[flip*I,blocks[1]-flip*I]])
 F=B.T.copy();F[0]=1;rhs=np.zeros(2*size);rhs[0]=1
 prob=solve(F,rhs)
 a=np.r_[marks[0].sum(axis=1),marks[1].sum(axis=1)];b=np.r_[squared[0],squared[1]]
 mean=prob@a
 P=B.copy();pivot=int(np.argmax(abs(prob)));P[pivot]=prob;rhs=mean-a;rhs[pivot]=0
 phi=solve(P,rhs)
 correction=np.r_[marks[0]@phi[:size],marks[1]@phi[size:]]
 noise=prob@(b+2*correction)
 return {'mean':float(mean),'noise':float(noise),'shot':float(prob@b),'generator_constant_error':float(max(abs(B@np.ones(2*size)))),
         'stationary_residual':float(max(abs(B.T@prob))),'poisson_residual':float(max(abs(B@phi-(mean-a)))),
         'motor':float(prob@np.r_[motors[0],motors[1]]),'velocity_square':float(prob@np.tile(x[1:-1]**2,2)),
         'conditional_mass_error':float(abs(prob[:size].sum()-.5))}


TRUE_PARAMETER=.67
INPUT_FIELDS=('angular_speed','flip_rate','temperature_a','temperature_b','flux_fraction')


def experiment(omega=0.,flip=1.,ta=1.,tb=1.,eta=1.,statistic='noise'):
    return dict(angular_speed=omega,flip_rate=flip,temperature_a=ta,temperature_b=tb,flux_fraction=eta,statistic=statistic)


def hidden_inputs():
    return {
        'equilibrium_flux_noise':[experiment(0.,flip,temp,temp,1.) for flip,temp in [(.8,.7),(1.,1.),(1.5,1.2),(2.,1.4)]],
        'temperature_flux_noise':[experiment(0.,flip,ta,tb,1.) for flip,ta,tb in [(.8,.6,1.4),(1.,1.6,.6),(1.5,1.4,.8),(2.,.8,1.2)]],
        'rotating_flux_noise':[experiment(omega,flip,ta,tb,1.) for omega,flip,ta,tb in [(.5,.8,1.,1.),(-.5,1.,.6,1.4),(.65,1.5,1.6,.6),(-.65,2.,.8,1.2)]],
        'maxwell_anchor':[experiment(omega,flip,ta,tb,0.,statistic) for omega,flip,ta,tb,statistic in [(0.,.8,1.4,.8,'mean'),(.5,1.,1.,1.,'noise'),(-.65,2.,.8,1.2,'noise'),(.4,1.5,1.3,.9,'mean')]],
    }


def predict(experiments,drag=TRUE_PARAMETER,points=121,extent=9.,quadrature=140):
    return np.array([state(float(drag),e['temperature_a'],e['temperature_b'],e['angular_speed'],e['flip_rate'],e['flux_fraction'],points,extent,quadrature)[e.get('statistic','mean')] for e in experiments])


def calibration_inputs():
    settings=[(ta,tb,flip) for ta,tb in [(1.5,.7),(.7,1.3),(1.4,.8),(.8,1.2)] for flip in [.8,1.2,2.]]
    return [dict(angular_speed=0.,flip_rate=flip,temperature_a=ta,temperature_b=tb,flux_fraction=0.) for _ in range(24) for ta,tb,flip in settings]
