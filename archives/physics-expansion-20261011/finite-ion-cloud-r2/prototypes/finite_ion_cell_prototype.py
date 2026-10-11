import numpy as np
from scipy.integrate import quad
from scipy.special import hyp2f1,exprel

def pressure(gamma,h,angular=True):
 L=np.log(h);a=4-3*gamma;b=2-gamma
 H=lambda d:hyp2f1(-gamma/2,-gamma/2,1,np.exp(-2*d)) if angular else 1.
 Z=quad(lambda d:H(d)*np.exp(b*d)*(L-d)*exprel(a*(L-d)),0,L,epsabs=1e-11,epsrel=1e-11)[0]
 P=quad(lambda d:H(d)*np.exp(b*d+a*(L-d)),0,L,epsabs=1e-11,epsrel=1e-11)[0]
 return -np.expm1(-2*L)*P/(4*Z)

if __name__=='__main__':
 for gamma in [.3,1.,1.5,2.,3.,4.]:
  for h in [1.25,1.5,2.,3.,5.,12.]:
   a=pressure(gamma,h);b=pressure(gamma,h,False)
   print(gamma,h,a,b,(b-a)/a)
