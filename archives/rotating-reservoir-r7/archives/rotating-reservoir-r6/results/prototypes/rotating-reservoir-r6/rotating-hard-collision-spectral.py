"""Independent Chebyshev collocation of the continuous rotor jump generator."""
import numpy as np
from scipy.linalg import solve
from numpy.polynomial.legendre import leggauss
import importlib.util,json,time

def interpolation(nodes,weights,targets):
 delta=targets[:,None]-nodes[None,:]
 near=abs(delta)<1e-13
 safe=np.where(near,1.,delta)
 value=weights[None,:]/safe
 value/=value.sum(axis=1)[:,None]
 for i in np.flatnonzero(near.any(axis=1)):
  value[i]=0;value[i,np.argmax(near[i])]=1
 return value

def state(gamma=.67,ta=1.,tb=1.,omega=.8,flip=1.,eta=1.,points=91,extent=9.,quadrature=100):
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
 nodes,weights=leggauss(quadrature);blocks=[];marks=[];squared=[];size=points-2
 for sign in [1.,-1.]:
  u=sign*omega;J=np.zeros((size,size));A=np.zeros_like(J);b=np.zeros(size)
  for i,v in enumerate(x[1:-1]):
   for lo,hi in [(-width,v),(v,width)]:
    g=(lo+hi)/2+(hi-lo)*nodes/2
    phi=np.exp(-(g-u)**2/(2*tb))/np.sqrt(2*np.pi*tb)
    rates=3.6*((1-eta)+eta*abs(v-g))*phi*weights*(hi-lo)/2
    mat=interpolation(x,w,g)@E
    q=.5*((v-u)**2-(g-u)**2)
    J[i]+=rates@mat;J[i,i]-=rates.sum()
    A[i]+=(rates*q)@mat;b[i]+=rates@q**2
  blocks.append(L[1:-1]+J);marks.append(A);squared.append(b)
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
 return {'mean':float(mean),'noise':float(noise),'shot':float(prob@b),'generator_constant_error':float(max(abs(B@np.ones(2*size))))}

if __name__=='__main__':
 spec=importlib.util.spec_from_file_location('fv','/private/tmp/rotating-hard-collision-prototype.py');fv=importlib.util.module_from_spec(spec);spec.loader.exec_module(fv)
 rows=[];start=time.time()
 for omega in [0,.5,1.5,3]:
  for ta,tb in [(1.,1.),(.6,1.4),(1.6,.6)]:
   e=dict(omega=omega,ta=ta,tb=tb,flip=.8)
   exact=state(**e);fine=state(**e,points=121,quadrature=140)
   coarse=fv.state(**e,points=401);fvfine=fv.state(**e,points=801)
   extrapolated={k:(4*fvfine[k]-coarse[k])/3 for k in ['mean','noise']}
   rows.append(dict(input=e,spectral=exact,refined=fine,independent_flux=extrapolated))
   print(e,'noise',fine['noise'],'refine',abs(exact['noise']-fine['noise']),'independent',abs(fine['noise']-extrapolated['noise']))
 report=dict(status='proposal_independent_reference',rows=rows,seconds=time.time()-start)
 open('/private/tmp/rotating-hard-collision-spectral.json','w').write(json.dumps(report,indent=2)+'\n')
 print('seconds',report['seconds'])
