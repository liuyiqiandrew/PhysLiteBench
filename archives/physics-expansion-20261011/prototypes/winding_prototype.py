import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln, hyp2f1, iv, roots_genlaguerre


def log_characteristic(s, time):
    s=abs(s)
    rho=np.exp(-time)
    return -time*s+2*gammaln(1+s/2)-gammaln(1+s)+np.log(hyp2f1(s/2,s/2,1+s,rho*rho))


def integral(s, time, order=80):
    # Rayleigh-weighted two-radius transition kernel. At long times divide
    # out rho^s, avoiding tiny characteristic values in the quadrature.
    s=abs(s);rho=np.exp(-time)
    x,w=roots_genlaguerre(order,s/2)
    z=2*rho*np.sqrt(x[:,None]*x[None,:])
    normalized=iv(s,z)/(rho*np.sqrt(x[:,None]*x[None,:]))**s
    return (1-rho*rho)*float(w@normalized@w)


if __name__=='__main__':
    for s in [.2,.5,1.,1.5,2.,3.]:
        v=integral(s,10)
        analytic=np.exp(log_characteristic(s,10)+s*10)
        rate=-(log_characteristic(s,14)-log_characteristic(s,10))/4
        print(s,v,analytic,rate,rate-s)
