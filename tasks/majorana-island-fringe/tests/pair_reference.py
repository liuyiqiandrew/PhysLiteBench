"""Spin matrices, two-photon wavefunctions, and direct thermal/voltage quadrature."""
from itertools import product

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.linalg import null_space

from cavity_reference import active_density


def polynomial(left,right,delta):
    """Normal-order the residual vacuum expectation after the displacement."""
    if not left:
        return np.prod([-np.vdot(delta,g) for g in right])
    f=left[0]
    value=np.vdot(f,delta)*polynomial(left[1:],right,delta)
    for j,g in enumerate(right):
        value+=np.vdot(f,g)*polynomial(left[1:],right[:j]+right[j+1:],delta)
    return value


def reference(q, thermal_order=11, voltage_order=20):
    cavity=q['cavity']
    src=cavity['pair_source']
    eta=cavity['transmission']
    n=len(eta)
    m=np.r_[0j,[complex(*f) for f in cavity['mode']],np.zeros(n)]
    m*=np.exp(1j*cavity['angle'])/np.linalg.norm(m)
    rest_frame=null_space(m.conj()[None,:]).conj().T
    def packet(raw):
        f=np.array([complex(*f) for f in raw])
        f/=np.linalg.norm(f)
        r=0j
        for j,t in enumerate(eta):
            incoming=f[j]
            f[j]=np.sqrt(t)*incoming-np.sqrt(1-t)*r
            r=np.sqrt(t)*r+np.sqrt(1-t)*incoming
        return np.r_[r,f]
    states=[]
    for arm in src['preparations']:
        f,g=map(packet,arm)
        norm=np.sqrt(1+abs(np.vdot(f,g))**2)
        a,b=np.vdot(m,f),np.vdot(m,g)
        fr,gr=rest_frame@f,rest_frame@g
        states.append([[(1/norm,[fr,gr])],[(a/norm,[gr]),(b/norm,[fr])],[(np.sqrt(2)*a*b/norm,[])]])
    c=complex(*src['pump'])
    s=complex(*src['marker'])
    mark=np.array([[1.,s],[s.conjugate(),1.]])
    amplitudes=[1,c]
    def rest_matrix(delta):
        out=np.zeros((3,3),complex)
        for i,first in enumerate(states):
            for j,second in enumerate(states):
                weight=np.conj(amplitudes[i])*amplitudes[j]*mark[i,j]
                for k,bra in enumerate(first):
                    for l,ket in enumerate(second):
                        for fa,fs in bra:
                            for ga,gs in ket:
                                out[k,l]+=weight*np.conj(fa)*ga*polynomial(fs,gs,delta)
        return out
    norm=np.trace(rest_matrix(np.zeros(2*n))).real
    rho,gammas=active_density(q)
    identity=np.eye(len(rho))
    u,weights=hermgauss(thermal_order)
    z0=np.sqrt(cavity['thermal'])*(u[:,None]+1j*u).ravel()
    thermal=(weights[:,None]*weights).ravel()/np.pi
    lo,hi=cavity['window']
    x,w=leggauss(voltage_order)
    x=(lo+hi)/2+(hi-lo)*x/2
    w*= (hi-lo)/2
    def paths(arm):
        outputs=[]
        for meter in (1,-1):
            for signs in product((1,-1),repeat=n):
                z=np.zeros((len(z0),2*n+1),complex)
                z[:,0]=z0
                phases=np.zeros(len(z0))
                spin=identity.copy()
                start=0
                for j,(stop,a,b,ax,ay) in enumerate(cavity['segments'][arm]):
                    for i,k,th,g in q['arms'][arm][start:stop]:
                        h=(th+meter*g)/2
                        spin=(np.cos(h)*identity-np.sin(h)*gammas[i]@gammas[k])@spin
                    start=stop
                    sig=signs[j]
                    spin=(identity-1j*sig*gammas[a]@gammas[b])@spin/2
                    kick=sig*complex(ax,ay)
                    phases+=np.imag(z[:,0].conj()*kick)
                    z[:,0]+=kick
                    z[:,j+1]=-np.sqrt(1-eta[j])*z[:,0]
                    z[:,0]*=np.sqrt(eta[j])
                if meter==-1:
                    spin*=q['outcome']*np.exp(-1j*q['meter_phase'])
                observed=z@m.conj()
                residual=z@rest_frame.T
                displacement=np.sqrt(2)*observed.real[:,None]
                center=x-displacement
                phase=phases[:,None]+np.sqrt(2)*observed.imag[:,None]*(x-displacement/2)
                gaussian=np.exp(-.5*center**2+1j*phase)/np.pi**.25
                psi=np.stack([gaussian,np.sqrt(2)*center*gaussian,(2*center**2-1)*gaussian/np.sqrt(2)],axis=1)
                outputs.append((spin/2,residual,psi))
        return outputs
    cache={}
    def overlap(left,right):
        value=0j
        for a,r,pa in left:
            for b,t,pb in right:
                spin=np.trace(a.conj().T@b@rho)
                if abs(spin)<1e-16:
                    continue
                delta=t[0]-r[0]
                key=tuple(delta.round(13))
                if key not in cache:
                    cache[key]=rest_matrix(delta)/norm
                vac=np.exp(-.5*np.vdot(delta,delta).real+1j*np.sum(r.conj()*t,axis=1).imag)
                spatial=(pa.conj()*w)@pb.swapaxes(1,2)
                kernel=np.einsum('mij,ij->m',spatial,cache[key])
                value+=spin*np.dot(thermal*vac,kernel)
        return value
    a,b=paths(0),paths(1)
    prob=(overlap(a,a)+overlap(b,b)).real/2
    return prob,(np.exp(-1j*q['phase'])*overlap(a,b)).real/prob
