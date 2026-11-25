import numpy as np
from scipy.linalg import eigh
from scipy.special import roots_legendre

C=np.pi**2/12+.25

def effective(e,modes=48,points=240,geometric=True):
    q,w=roots_legendre(points);x=(q+1)/2;w=w/2
    n=np.arange(1,modes+1)
    b=np.sqrt(2)*np.sin(np.pi*x[:,None]*n)
    ap=-2*np.pi*e['amplitude']*np.sin(2*np.pi*x)-4*np.pi*e['harmonic']*np.sin(4*np.pi*x+e['phase'])
    potential=e['well']*np.cos(2*np.pi*x)+(C*ap**2 if geometric else 0.)
    h=np.diag((np.pi*n)**2)+b.T@(w[:,None]*potential[:,None]*b)
    v=eigh(h,eigvals_only=True,subset_by_index=(0,e['level']))
    return v[e['level']]-v[0]

# Independent 2D Rayleigh-Ritz: normalized moving transverse basis,
# keeping all derivative couplings of the physical x,y gradient.
def finite(e,epsilon,modes=40,transverse=7,points=220):
    q,w=roots_legendre(points);x=(q+1)/2;w=w/2
    qt,wt=roots_legendre(100);u=qt/2;wt=wt/2
    n=np.arange(1,modes+1)
    phi=np.sqrt(2)*np.sin(np.pi*x[:,None]*n)
    dphi=np.sqrt(2)*np.pi*n*np.cos(np.pi*x[:,None]*n)
    t=np.arange(1,transverse+1)
    chi=np.sqrt(2)*np.sin(np.pi*(u[:,None]+.5)*t)
    dchi=np.sqrt(2)*np.pi*t*np.cos(np.pi*(u[:,None]+.5)*t)
    dilation=u[:,None]*dchi+.5*chi
    B=chi.T@(wt[:,None]*dilation)
    G=dilation.T@(wt[:,None]*dilation)
    a=np.exp(e['amplitude']*np.cos(2*np.pi*x)+e['harmonic']*np.cos(4*np.pi*x+e['phase']))
    r=-2*np.pi*e['amplitude']*np.sin(2*np.pi*x)-4*np.pi*e['harmonic']*np.sin(4*np.pi*x+e['phase'])
    grad=phi.T@((w*r)[:,None]*dphi)
    h=np.kron(np.diag((np.pi*n)**2),np.eye(transverse))-np.kron(grad.T,B)-np.kron(grad,B.T)
    h+=np.kron(phi.T@((w*r*r)[:,None]*phi),G)
    h+=np.kron(phi.T@((w/(epsilon*a)**2)[:,None]*phi),np.diag(np.pi**2*(t*t-1)))
    h+=np.kron(phi.T@((w*e['well']*np.cos(2*np.pi*x))[:,None]*phi),np.eye(transverse))
    v=eigh(h,eigvals_only=True,subset_by_index=(0,e['level']))
    return v[e['level']]-v[0]

if __name__=='__main__':
    for a,b in [(.4,0),(.8,0),(.4,.3),(0,.4),(.7,.25)]:
        e=dict(amplitude=a,harmonic=b,phase=.7,well=5.,level=1)
        print(e,'effective',effective(e),'shortcut',effective(e,geometric=False),'finite',[finite(e,ep) for ep in [.06,.03,.015,.0075]])
