import numpy as np,json
from scipy.special import spence
from pathlib import Path

def bands(m,hand=1,n=128):
 k=-np.pi+2*np.pi*(np.arange(n)+.5)/n
 x,y=np.meshgrid(k,k,indexing='ij')
 d=np.stack([np.sin(x),hand*np.sin(y),m+np.cos(x)+np.cos(y)],axis=-1)
 dx=np.stack([np.cos(x),np.zeros_like(x),-np.sin(x)],axis=-1)
 dy=np.stack([np.zeros_like(y),hand*np.cos(y),-np.sin(y)],axis=-1)
 r=np.linalg.norm(d,axis=-1)
 # Lower band has plus half solid-angle curvature with i<du|cross|du> convention.
 omega=np.sum(d*np.cross(dx,dy),axis=-1)/(2*r**3)
 return np.stack([4.5-r,4.5+r]),np.stack([omega,-omega])

def response(m,T,J=1,hand=1,n=128):
 e,o=bands(m,hand,n);e*=J;x=e/T;b=1/np.expm1(x);z=np.exp(-x)
 c2=x*x*b-2*x*np.log1p(-z)+2*spence(1-z)
 sigma=-np.mean(np.sum(o*b,axis=0));thermal=-T*np.mean(np.sum(o*c2,axis=0))
 kubo=(4.5*J)**2/T*sigma
 return sigma,thermal,kubo

def edge(T,e,o):
 x=e/T;z=np.exp(-x)
 integrated_energy=T*T*(spence(1-z)-x*np.log1p(-z))
 return np.mean(np.sum(o*integrated_energy,axis=0))
rows=[];checks=[]
for m in [-1.5,-1.,-.6]:
 for T in [.4,.7,1.1,1.5]:
  p,q,s=response(m,T);e,o=bands(m);h=T*.001
  ref=-(edge(T-2*h,e,o)-8*edge(T-h,e,o)+8*edge(T+h,e,o)-edge(T+2*h,e,o))/(12*h)
  checks.append(abs(ref-q));rows.append(dict(m=m,T=T,particle=p,true_thermal=q,kubo_only=s,relative_error=abs((s-q)/q)))
report=dict(rows=rows,max_edge_derivative_error=max(checks),primary_source='https://arxiv.org/html/1106.1987',status='Prototype only; apparatus and independent Berry reference require peer review.')
Path('results/boson-hall-prototype.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
