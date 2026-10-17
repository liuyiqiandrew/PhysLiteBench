"""Read-only proposal: fixed-radius rotor heated by a flux-weighted rotating gas."""
import numpy as np
from scipy.linalg import solve
from scipy.special import ndtr
import json,time


def bernoulli(x):
 out=np.empty_like(x);small=abs(x)<1e-5
 out[small]=1-x[small]/2+x[small]**2/12-x[small]**4/720
 out[~small]=x[~small]/np.expm1(x[~small]);return out

def state(gamma=.67,ta=1.,tb=1.,omega=.8,flip=1.,eta=1.,points=241,extent=9.,closure=False):
 # Equal effective mass, unit radius. The signed angular speed v swaps with
 # incoming gas speed g; outgoing gas heat in its rotating frame is q.
 width=extent*np.sqrt(max(ta,tb))+abs(omega)
 edges=np.linspace(-width,width,points+1);x=(edges[1:]+edges[:-1])/2;h=edges[1]-edges[0]
 v=x[:,None];g=x[None,:];lam=3.6
 D=gamma*ta;drift=-gamma*edges[1:-1]
 pu=D/h**2*bernoulli(-drift*h/D);pd=D/h**2*bernoulli(drift*h/D)
 L=np.diag(pu,1)+np.diag(pd,-1)
 L-=np.diag(L.sum(axis=1))
 blocks=[];J1=[];J2=[]
 for sign in [1.,-1.]:
  u=sign*omega
  gas=np.exp(-(x-u)**2/(2*tb))/np.sqrt(2*np.pi*tb)*h
  rate=lam*((1-eta)+eta*(np.sqrt(4*tb/np.pi) if closure else abs(v-g)))
  if np.ndim(rate)==0:rate=np.full((points,points),rate)
  jump=rate*gas[None,:]
  heat=.5*((v-u)**2-(g-u)**2)
  J1.append(jump*heat);J2.append(jump*heat**2)
  blocks.append(L+jump-np.diag(jump.sum(axis=1)))
 eye=np.eye(points)
 B=np.block([[blocks[0]-flip*eye,flip*eye],[flip*eye,blocks[1]-flip*eye]])
 F=B.T.copy();F[0]=1;rhs=np.zeros(2*points);rhs[0]=1
 prob=solve(F,rhs)
 a=np.r_[J1[0].sum(axis=1),J1[1].sum(axis=1)]
 b=np.r_[J2[0].sum(axis=1),J2[1].sum(axis=1)]
 mean=prob@a
 system=B.copy();pivot=int(np.argmax(prob));system[pivot]=prob;rhs=mean-a;rhs[pivot]=0
 corrector=solve(system,rhs)
 correlation=np.r_[J1[0]@corrector[:points],J1[1]@corrector[points:]]
 noise=prob@(b+2*correlation)
 return {'mean':float(mean),'noise':float(noise),'shot':float(prob@b),'min_probability':float(prob.min()),'edge_probability':float(prob[[0,points-1,points,2*points-1]].sum()),'normalization_error':float(abs(prob.sum()-1)), 'collision_noise_correlation':float(2*prob@correlation)}

if __name__=='__main__':
 start=time.time();rows=[]
 for omega in [0,.5,1.5,3.]:
  for ta,tb in [(1.,1.),(.6,1.4),(1.6,.6)]:
   e=dict(omega=omega,ta=ta,tb=tb,flip=.8)
   exact=state(**e);source=state(**e,closure=True);refined=state(**e,points=401)
   rows.append(dict(input=e,exact=exact,shortcut=source,refined=refined,noise_gap=abs(exact['noise']-source['noise'])/exact['noise']))
 report=dict(status='proposal_only',assumptions='unit-radius fixed-ring equal-effective-mass elastic exchange, stationary rotating hard-collision gas plus lab Langevin thermostat',rows=rows,seconds=time.time()-start)
 open('/private/tmp/rotating-hard-collision-prototype.json','w').write(json.dumps(report,indent=2)+'\n')
 for r in rows:print(r['input'],'noise gap',r['noise_gap'],'mean',r['exact']['mean'],'refine',abs(r['refined']['noise']-r['exact']['noise']))
 print('seconds',report['seconds'])
