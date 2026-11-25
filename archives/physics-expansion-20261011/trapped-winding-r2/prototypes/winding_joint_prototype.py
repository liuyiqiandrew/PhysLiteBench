import numpy as np
from scipy.special import iv,roots_genlaguerre


def log_characteristic(probe, radial, mu, k, temperature, time, order=64):
    a=mu*k; D=mu*temperature;nu=abs(probe)
    omega=np.sqrt(a*a+4*D*radial)
    tau=omega*time
    coefficient=(a+omega/np.tanh(tau))/(4*D)
    z=omega/(2*D*np.sinh(tau)*coefficient)
    x,w=roots_genlaguerre(order,nu/2)
    p=z*np.sqrt(x[:,None]*x[None,:])/2
    normalized=iv(nu,2*p)/p**nu
    integral=float(w@normalized@w)
    return a*time+np.log(a*omega/(8*D*D*coefficient**2))-np.log(np.sinh(tau))+nu*np.log(z/2)+np.log(integral)


if __name__=='__main__':
    for nu,s in [(0,.7),(.5,0),(.5,.7),(1.5,1.2),(2.,1.8)]:
        mu,k,T=1.09,.8,1.5
        a=mu*k;D=mu*T;omega=np.sqrt(a*a+4*D*s)
        t1,t2=10/omega,14/omega
        numeric=-(log_characteristic(nu,s,mu,k,T,t2)-log_characteristic(nu,s,mu,k,T,t1))/(t2-t1)
        exact=omega*(1+nu)-a
        independent=a*nu+omega-a
        print(nu,s,numeric,exact,numeric-exact,independent)
