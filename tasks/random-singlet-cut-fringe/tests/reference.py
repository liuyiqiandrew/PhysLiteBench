"""Solve the marked-link density flow with explicit count polynomials."""
import numpy as np
from scipy.integrate import solve_ivp


def spin_weights(band,side):
    I=np.eye(2,dtype=complex)
    axes={'x':np.array([[0,1],[1,0]]),'y':np.array([[0,-1j],[1j,0]]),'z':np.diag([1,-1])}
    eps=np.array([[0.,1.],[-1.,0.]])/np.sqrt(2)
    def pair(arm):
        ends=[I,I]
        for which,axis,mag,gradient in arm:
            k=int(which=='R')
            angle=(mag+gradient*side*(-1)**k)/2
            ends[k]=(np.cos(angle)*I-1j*np.sin(angle)*axes[axis])@ends[k]
        return ends[0]@eps@ends[1].T
    m=np.vdot(pair(band['arms'][0]),pair(band['arms'][1]))
    gain,mag,diff=band['herald']
    p=gain*(1-np.cos(mag+diff*side))/2
    return m,p


def reference(q):
    n=1 if q['count'] is None else q['count']+1
    # Rows of coefficients in (a+b*x)*exp(-x) per next sublattice, count and readout channel.
    y=np.zeros((2,n,2,2),complex)
    for i,side in enumerate((1,-1)):
        p=.5 if q['last'] is None else float(q['last']==-side)
        y[:,:,i,:][:,0]=p*np.array([0.,1.] if q['start']=='crossing' else [1/3,2/3])
    shape=y.shape
    for band in q['bands']:
        factors=[spin_weights(band,s) for s in (1,-1)]
        def rhs(t,packed):
            y=packed.reshape(shape)
            a,b=y[...,0],y[...,1]
            d=np.stack([b-2*a,a-b],axis=-1)
            for i,(fringe,prob) in enumerate(factors):
                feedback=a[:,:,i]*np.array([1,fringe])[:,None]
                hit=0. if q['count'] is None else prob
                d[:,:,1-i,1]+=feedback*(1-hit)
                d[:,1:,1-i,1]+=feedback[:,:-1]*hit
            return d.ravel()
        ans=solve_ivp(rhs,(0,band['width']),y.ravel(),rtol=2e-11,atol=1e-13,method='DOP853')
        y=ans.y[:,-1].reshape(shape)
    rate,overlap=y[:,-1].sum(axis=(1,2))
    return rate.real,(np.exp(-1j*q['phase'])*overlap).real/rate.real
