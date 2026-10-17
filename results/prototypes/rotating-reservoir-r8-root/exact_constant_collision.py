"""Independent exact moment/Poisson solution at velocity-independent collision rate.
Full telegraph-driven finite gas and rotor; no Gaussian stationary closure.
"""
import numpy as np

def add(polynomial,key,value):polynomial[key]=polynomial.get(key,0.)+value

def generator(key,gamma,lam,ta,tb,omega,kappa,nu):
 a,b,r=key;out={};add(out,key,-a*gamma-b*lam-2*kappa*r-nu);add(out,(b,a,r),nu)
 if a>=2:add(out,(a-2,b,r),a*(a-1)*gamma*ta)
 if b>=2:add(out,(a,b-2,r),b*(b-1)*lam*tb)
 if b:add(out,(a,b-1,1-r),b*lam*omega)
 return out

def multiply(first,second):
 out={}
 for (a,b,r),v in first.items():
  for (c,d,s),w in second.items():add(out,(a+c,b+d,(r+s)%2),v*w)
 return out

def exact(ta,tb,gamma=.67,lam=2.,omega=.6,kappa=.8,nu=3.6):
 keys=[(a,d-a,r) for d in range(5) for a in range(d+1) for r in range(2)];index={k:i for i,k in enumerate(keys)};matrix=np.zeros((len(keys),len(keys)))
 for j,k in enumerate(keys):
  for power,value in generator(k,gamma,lam,ta,tb,omega,kappa,nu).items():matrix[index[power],j]+=value
 constraints=matrix.T.copy();constraints[0]=0;constraints[0,0]=1;rhs=np.zeros(len(keys));rhs[0]=1;moments=np.linalg.solve(constraints,rhs)
 def expect(poly):return float(sum(value*moments[index[key]] for key,value in poly.items()))
 zero=(0,0,0);heat={(0,2,0):lam,(0,1,1):-2*lam*omega,zero:lam*(omega**2-tb)};mean=expect(heat);centered=dict(heat);centered[zero]-=mean
 small=[i for i,key in enumerate(keys) if key[0]+key[1]<=2];L=matrix[np.ix_(small,small)].copy();forcing=-np.array([centered.get(keys[i],0.) for i in small]);L[0]=0;L[0,0]=1;forcing[0]=0;coeff=np.linalg.solve(L,forcing);phi={keys[i]:v for i,v in zip(small,coeff)}
 derivative={(a,b-1,r):b*v for (a,b,r),v in phi.items() if b};rel={(0,1,0):1.,(0,0,1):-omega};rel2=multiply(rel,rel)
 direct=2*lam*tb*expect(rel2);cross=-4*lam*tb*expect(multiply(rel,derivative));correlation=2*expect(multiply(centered,phi));noise=direct+cross+correlation
 covariance=moments[index[(1,1,0)]];v2=moments[index[(2,0,0)]];g2=moments[index[(0,2,0)]];sg=moments[index[(0,1,1)]];work=lam*(omega**2-omega*sg);ja=gamma*(v2-ta)
 assert abs(ja+mean-work)<2e-12 and noise>=-2e-12
 return dict(mean=mean,noise=noise,direct_noise=direct,shared_noise_cross=cross,correlation=correlation,energy_residual=ja+mean-work,moment_stationarity_residual=float(max(abs(matrix.T@moments))),moments={str(key):float(moments[i]) for key,i in index.items()})

if __name__=='__main__':
 import itertools,json
 from pathlib import Path
 rows=[]
 for ta,tb,omega,lam in itertools.product([.4,1.,2.],[.4,1.4],[0.,.6],[.2,2.,3.]):
  r=exact(ta,tb,lam=lam,omega=omega);r['input']=dict(temperature_a=ta,temperature_b=tb,angular_speed=omega,relaxation_rate=lam,flip_rate=.8,flux_fraction=0.);rows.append(r)
 Path(__file__).with_suffix('.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n');print('72 exactmoments/noise anchors; maxmomentresidual',max(r['moment_stationarity_residual'] for r in rows));print('example',exact(1.6,.4,lam=2.,omega=.6)['noise'])
