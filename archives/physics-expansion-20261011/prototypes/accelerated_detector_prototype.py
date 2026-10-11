import numpy as np
from scipy.integrate import quad


def spectrum(omega, acceleration, separation):
    a=acceleration; L=separation
    eta=min(1/abs(omega),np.pi/a) if a else 1/abs(omega)
    def wightman(u):
        z=u-1j*eta
        if a and a*u>700:return 0j
        interval=4*np.sinh(a*z/2)**2/a**2 if a else z*z
        return -1/(4*np.pi**2*(interval-L*L))
    c,ce=quad(lambda u:wightman(u).real,0,np.inf,weight='cos',wvar=omega,epsabs=1e-11,limlst=160,limit=250)
    s,se=quad(lambda u:wightman(u).imag,0,np.inf,weight='sin',wvar=omega,epsabs=1e-11,limlst=160,limit=250)
    return 2*np.exp(omega*eta)*(c-s)


def analytic(omega,a,L):
    local=omega/(2*np.pi)/(-np.expm1(-2*np.pi*omega/a)) if a else omega/(2*np.pi)
    if not L:return local
    f=np.sin(2*omega/a*np.arcsinh(a*L/2))/(omega*L*np.sqrt(1+(a*L/2)**2)) if a else np.sinc(omega*L/np.pi)
    return local*f

if __name__=='__main__':
    for a in [0.,.01,.5,3.,10.]:
        for om in [.5,2.]:
            for L in [0.,.1,1.,3.]:
                numeric=spectrum(om,a,L);exact=analytic(om,a,L)
                print(a,om,L,numeric,exact,abs(numeric-exact))
