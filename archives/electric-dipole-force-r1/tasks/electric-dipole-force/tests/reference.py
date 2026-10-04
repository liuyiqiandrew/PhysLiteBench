"""Independent incident-plus-radiated Maxwell-stress surface integral."""
import numpy as np
from numpy.polynomial.legendre import leggauss

def beam(theta,te=0j,tm=1+0j,phase=0.):
 k=np.array([np.sin(theta),0,np.cos(theta)])
 e=(te*np.array([0.,1.,0.])+tm*np.array([np.cos(theta),0,-np.sin(theta)]))*np.exp(1j*phase)
 return dict(direction=k.tolist(),real=e.real.tolist(),imag=e.imag.tolist())
def fields(waves,points):
 points=np.asarray(points);E=np.zeros(points.shape,complex);H=np.zeros_like(E)
 for w in waves:
  k=np.array(w['direction']);e=np.array(w['real'])+1j*np.array(w['imag']);phase=np.exp(1j*(points@k))
  E+=phase[...,None]*e;H+=phase[...,None]*np.cross(k,e)
 return E,H

def stress_force(e,s=1.1,radius=.4,order=24):
 z,w=leggauss(order);phi=np.arange(2*order)*np.pi/order
 zz,pp=np.meshgrid(z,phi,indexing='ij');nn=np.stack([np.sqrt(1-zz*zz)*np.cos(pp),np.sqrt(1-zz*zz)*np.sin(pp),zz],axis=-1).reshape(-1,3)
 weights=np.repeat(w,2*order)*np.pi/order*radius**2
 E,H=fields(e['waves'],np.array(e['position'])+radius*nn);center,_=fields(e['waves'],np.array(e['position']))
 a=s/(1-1j*s/(6*np.pi));p=a*center;proj=(nn@p)[:,None]*nn;scale=np.exp(1j*radius)/(4*np.pi)
 Es=scale*((p-proj)/radius+(3*proj-p)*(1/radius**3-1j/radius**2))
 Hs=scale*(1/radius+1j/radius**2)*np.cross(nn,p)
 def traction(E,H):
  return .5*np.real(E*np.sum(E.conj()*nn,axis=1)[:,None]+H*np.sum(H.conj()*nn,axis=1)[:,None]-.5*nn*(np.sum(abs(E)**2+abs(H)**2,axis=1))[:,None])
 return weights@(traction(E+Es,H+Hs)-traction(E,H))

TRUE_PARAMETER=1.1


def experiment(waves,position=(0.,0.,0.),axis=(0.,0.,1.)):
 return dict(waves=waves,position=list(position),axis=list(axis))


def calibration_inputs():
 result=[]
 for theta in [0.,.3,.7]:
  for te,tm in [(1.,0.),(0.,1.),(.6,.8),(.6j,.8)]:
   for amplitude in [.5,.75,1.]:
    wave=beam(theta,amplitude*te,amplitude*tm)
    result.append(experiment([wave],(.17,-.11,.31),wave['direction']))
 for theta in [.25,.5,.8]:
  for phase in [.4,1.,1.6,2.2]:
   for x in [-.4,0.,.4]:
    for axis in [(1.,0.,0.),(0.,0.,1.),(.6,0.,.8)]:
     result.append(experiment([beam(theta,1.,0.),beam(-theta,1.,0.,phase)],(x,.13,-.19),axis))
 return result


def hidden_inputs():
 groups={'tm_interference':[],'mixed_polarization':[],'rotated_geometry':[]}
 for theta,phase,x in [(.55,.3,.17),(.73,.8,-.11),(.91,1.1,.23)]:
  for axis in [(0.,0.,1.),(0.,.6,.8),(.06,0.,np.sqrt(1-.06**2))]:
   groups['tm_interference'].append(experiment([beam(theta),beam(-theta,phase=phase)],(x,.21,.31),axis))
 for theta,phase,x in [(.62,.27,-.16),(.81,.73,.19),(.96,1.17,-.23)]:
  for axis in [(0.,0.,1.),(0.,.6,.8),(.04,.3,np.sqrt(1-.04**2-.3**2))]:
   groups['mixed_polarization'].append(experiment([beam(theta,.3j,.8),beam(-theta,.2,.9j,phase)],(x,-.24,.13),axis))
 for theta,phase,rotation in [(.67,.4,.37),(.83,.6,.91),(.97,.9,1.43)]:
  c,s=np.cos(rotation),np.sin(rotation);R=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
  for axis in [(0.,0.,1.),(0.,.4,np.sqrt(.84)),(.03,.2,np.sqrt(1-.03**2-.2**2))]:
   waves=[beam(theta,.2,.9),beam(-theta,.1j,.8,phase)]
   for w in waves:
    for key in ['direction','real','imag']:w[key]=(R@w[key]).tolist()
   groups['rotated_geometry'].append(experiment(waves,R@np.array([.12,-.17,.23]),R@axis))
 return groups


def predict(experiments,response_strength=TRUE_PARAMETER,radius=.4,order=24):
 return np.array([np.array(e['axis'])@stress_force(e,response_strength,radius,order) for e in experiments])
