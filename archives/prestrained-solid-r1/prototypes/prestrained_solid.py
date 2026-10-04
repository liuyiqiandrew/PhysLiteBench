import itertools,json
import numpy as np
from scipy.optimize import brentq

def matrices(F,n):
 C=F.T@F;inv=np.linalg.inv(C);I=np.eye(3);L=np.log(np.linalg.det(F));x=2*L-1.4
 S=.6*I+.4*(np.trace(C)*I-C)+x*inv
 CC=.8*(np.einsum('ab,cd->abcd',I,I)-.5*(np.einsum('ac,bd->abcd',I,I)+np.einsum('ad,bc->abcd',I,I)))+2*np.einsum('ab,cd->abcd',inv,inv)-x*(np.einsum('ac,bd->abcd',inv,inv)+np.einsum('ad,bc->abcd',inv,inv))
 v=F.T@n
 Q=np.einsum('ib,a,bacd,c,jd->ij',F,v,CC,v,F)
 exact=Q+float(v@S@v)*I
 return Q,exact,S

def lateral(b):
 return brentq(lambda a:matrices(np.diag([a,a,b]),np.array([1.,0.,0.]))[2][0,0],.5,1.2,xtol=1e-14)

if __name__=='__main__':
 out={}
 for b in [1.,1.2,1.4,1.6]:
  a=lateral(b);F=np.diag([a,a,b]);Q,E,S=matrices(F,np.array([1.,0.,0.]));print('cal',a,b,np.max(abs(Q-E)))
  Q,E,S=matrices(F,np.array([0.,0.,1.]));print('axial',b,np.sqrt(np.linalg.eigvalsh(Q)/np.linalg.eigvalsh(E))-1)
 minq=1e9;mine=1e9;cases=0
 for vals in itertools.product(np.linspace(.85,1.6,7),repeat=3):
  if not 1-1e-10<=np.prod(vals)<=1.5+1e-10:continue
  for n in [np.array([1.,0,0]),np.array([1.,1.,1.])/np.sqrt(3),np.array([.3,-.7,.6])/np.sqrt(.94)]:
   Q,E,_=matrices(np.diag(vals),n);minq=min(minq,float(np.linalg.eigvalsh(Q)[0]));mine=min(mine,float(np.linalg.eigvalsh(E)[0]));cases+=1
 print('positive',minq,mine,cases)
 for vals in [[1.1,1.1,1.1],[1.15,1.1,1.15],[1.3,1.05,1.05],[1.5,.95,1.02]]:
  for n in [np.array([1.,0,0]),np.array([1.,1.,1.])/np.sqrt(3)]:
   Q,E,S=matrices(np.diag(vals),n);print(vals,n.tolist(),np.sqrt(np.linalg.eigvalsh(Q)/np.linalg.eigvalsh(E))-1)
