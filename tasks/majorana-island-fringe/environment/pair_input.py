"""Ramsey fringe for two tagged wave-packet emissions and a homodyne voltage window."""
from functools import lru_cache

import numpy as np
from scipy.special import ndtr

from cavity import spin_function, paths


def pair_states(c, coherent):
    eta=c['transmission']
    n=len(eta)
    transform=np.eye(2*n+1)
    for i,e in enumerate(eta):
        h,j=transform[[0,i+1]].copy()
        transform[0]=np.sqrt(e)*h+np.sqrt(1-e)*j
        transform[i+1]=-np.sqrt(1-e)*h+np.sqrt(e)*j
    source=c['pair_source']
    matrices=[]
    for branch in source['preparations']:
        fs=[]
        for raw in branch:
            values=np.asarray(raw)
            f=np.r_[0.,values[:,0]+1j*values[:,1]]
            fs.append(transform@(f/np.linalg.norm(f)))
        f,g=fs
        matrices.append((np.outer(f,g)+np.outer(g,f))/np.sqrt(2*(1+abs(np.vdot(f,g))**2)))
    a=complex(*source['pump'])
    marker=complex(*source['marker']) if coherent else 0.
    weight=np.array([[1.,a*marker],[np.conj(a*marker),abs(a)**2]])
    norm=sum(weight[i,j]*np.vdot(si,sj) for i,si in enumerate(matrices)
             for j,sj in enumerate(matrices)).real
    return transform[:,0],np.array(matrices),weight/norm


def pair_fringe(q, coherent=True):
    c=q['cavity']
    bins=len(c['transmission'])
    v,states,weight=pair_states(c,coherent)
    g=np.r_[0.,[complex(*w) for w in c['mode']],np.zeros(bins)]
    g=g*np.exp(1j*c['angle'])/np.linalg.norm(g)
    M=np.eye(len(v))+2*c['thermal']*np.outer(v,v)
    sigma=np.sqrt(np.vdot(g,M@g).real/2)
    lo,hi=c['window']
    spin=spin_function(q,q['parity'])
    q1=-1j*g.conj()/np.sqrt(2)
    gram=np.einsum('iab,jab->ij',states.conj(),states)

    @lru_cache(None)
    def coefficients(encoded):
        delta=np.r_[encoded,np.zeros(bins)].conj()
        u=np.stack((states@delta,states@q1),axis=1)
        b=np.stack((u[:,0]@delta,u[:,0]@q1+u[:,1]@delta,u[:,1]@q1),axis=1)
        poly=np.zeros(5,complex)
        poly[0]=np.sum(weight*gram)
        for i in range(2):
            for j in range(2):
                for p in range(2):
                    for r in range(2):
                        poly[p+r]-=2*weight[i,j]*np.vdot(u[i,p],u[j,r])
                poly+=.5*weight[i,j]*np.convolve(b[i].conj(),b[j])
        return poly

    @lru_cache(None)
    def optical(a,b):
        delta0=np.subtract(b,a)
        sum0=np.add(a,b)
        delta=np.r_[delta0,np.zeros(bins)]
        mu=np.vdot(g[:bins+1],sum0).real/np.sqrt(2)+1j*np.vdot(g,M@delta).imag/np.sqrt(2)
        base=np.exp(1j*np.vdot(a,b).imag-.5*np.vdot(delta,M@delta).real)
        z=(np.array([lo,hi])-mu)/sigma
        normal=np.exp(-.5*z**2)/(np.sqrt(2*np.pi)*sigma)
        herm=[np.ones(2),z,z*z-1,z*(z*z-3)]
        moments=[ndtr(z[1])-ndtr(z[0])]
        for k,h in enumerate(herm):
            edge=normal*h*(-1/sigma)**k
            moments.append(1j**(k+1)*(edge[1]-edge[0]))
        return base*np.dot(coefficients(tuple(delta0)),moments)

    def overlap(left,right):
        val=0j
        for a,d,fa in left:
            for b,e,fb in right:
                val+=a.conjugate()*b*spin(fa,fb)*optical(tuple(d),tuple(e))
        return val
    a,b=paths(q,0),paths(q,1)
    prob=(overlap(a,a)+overlap(b,b)).real/2
    return prob,(np.exp(-1j*q['phase'])*overlap(a,b)).real/prob
