import numpy as np
from scipy.special import roots_legendre
from functools import lru_cache

@lru_cache(maxsize=None)
def nodes(n):
 x,w=roots_legendre(n);return (x+1)/2,w/2


def pressure(gamma,count,ratio,order=96):
 L=np.log(ratio);x,w=nodes(order)
 if count==1:
  Z=L*np.dot(w,np.exp((2-gamma)*L*x))
  return -np.expm1(-2*L)*np.exp((2-gamma)*L)/(2*Z)
 d=L*x;theta=np.pi*x
 factor=(1-np.exp(-d))[:,None]**2+4*np.exp(-d)[:,None]*np.sin(theta[None,:]/2)**2
 angular=(factor**(gamma/2))@w
 s=(L-d)[:,None]*x[None,:]
 radial=((L-d)*np.exp((2-gamma)*d))*((np.exp((4-3*gamma)*s))@w)
 Z=L*np.dot(w,angular*radial)
 boundary=L*np.dot(w,angular*np.exp((4-3*gamma)*(L-d)+(2-gamma)*d))
 return -np.expm1(-2*L)*boundary/(2*count*Z)

if __name__=='__main__':
 from finite_ion_cell_prototype import pressure as exact
 for gamma in [.07,.3,1.,1.5,2.,3.,4.2]:
  for ratio in [1.25,2,5,12]:
   p=exact(gamma,ratio)
   print(gamma,ratio,p,*[abs(pressure(gamma,2,ratio,n)-p) for n in [48,96,160]])
