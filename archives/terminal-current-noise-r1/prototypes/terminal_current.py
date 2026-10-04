import numpy as np
from scipy.special import expit

def spectrum(c=.6,omega=3.,offset=.1,left=-.6,right=.4,t=.2,gl=1.3,gr=.7,rate=1.1,correct=True):
 level=offset-c*left-(1-c)*right
 a=rate*np.array([gl,gr])*expit(-(level+np.array([left,right]))/t)
 b=rate*np.array([gl,gr])-a
 alpha,beta=(1-c,c) if correct else (1.,0.)
 entry=a.sum();exit=b.sum();s=entry+exit
 stationary=np.array([exit,entry])/s
 L=np.array([[-entry,exit],[entry,-exit]])
 J=np.array([[0.,-alpha*b[0]+beta*b[1]],[alpha*a[0]-beta*a[1],0.]])
 J2=np.array([[0.,alpha*alpha*b[0]+beta*beta*b[1]],[alpha*alpha*a[0]+beta*beta*a[1],0.]])
 P=np.outer(stationary,np.ones(2));Q=np.eye(2)-P
 R=np.linalg.solve(1j*omega*np.eye(2)-L+P,Q)
 return float(2*np.ones(2)@J2@stationary+4*np.real(np.ones(2)@J@R@J@stationary))

if __name__=='__main__':
 for w in [0.,.5,1.5,3.,6.,10.]:
  for c in [.2,.5,.8]:
   a=spectrum(c=c,omega=w);b=spectrum(c=c,omega=w,correct=False)
   print(c,w,a,b,b/a-1)
