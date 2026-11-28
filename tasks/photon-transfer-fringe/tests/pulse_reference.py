"""Real Hamilton equations and independent fugacity selection of one photon per trigger."""
import numpy as np
from scipy.linalg import expm


def passive(n, elements):
    out = np.eye(n,dtype=complex)
    for i,j,t,p in elements:
        rows = out[[i,j]].copy()
        out[i] = np.cos(t)*rows[0] - np.exp(-1j*p)*np.sin(t)*rows[1]
        out[j] = np.exp(1j*p)*np.sin(t)*rows[0] + np.cos(t)*rows[1]
    return out


def real_pulse(n, rail, row, duration):
    i,j,theta,phase,beta,f = row
    k = 2*n+2
    mode = np.zeros(k,complex)
    mode[[i,j]] = [np.cos(theta),np.exp(1j*phase)*np.sin(theta)]
    u,v = np.cos(beta)*mode,np.exp(1j*f)*np.sin(beta)*mode.conj()
    h = 2*n+rail
    H = np.zeros((2*k,2*k))
    H[h] = np.r_[(u+v).real,-(u-v).imag]
    H[h+k] = np.r_[(u+v).imag,(u-v).real]
    H += H.T.copy()
    J = np.block([[np.zeros((k,k)),np.eye(k)],[-np.eye(k),np.zeros((k,k))]])
    evolution=expm(duration * J @ H)
    A,B = np.hsplit(evolution[:k],2)
    C,D = np.hsplit(evolution[k:],2)
    return (A+D+1j*(C-B))/2,(A-D+1j*(C+B))/2


def finite_reference(q, points=72, radius=.012):
    n = len(q['source'])
    k = 2*n+2
    eye = np.eye(k)
    src = np.array(q['source'])
    r,phi = src[:,1],src[:,2]
    # Purify each thermal source by a two-mode squeezing operation on vacuum.
    # The physical single-mode squeeze follows the thermal preparation.
    base=np.zeros((k,k),complex)
    t = np.sqrt(src[:,0]/(1+src[:,0]))
    base[np.arange(n),n+np.arange(n)] = t
    base[n+np.arange(n),np.arange(n)] = t
    X = eye.copy()
    X[np.arange(n),np.arange(n)] = np.cosh(r)
    Y = np.zeros_like(base)
    Y[np.arange(n),np.arange(n)] = -np.exp(1j*phi)*np.sinh(r)
    divisor = X.conj()+Y.conj() @ base
    base=(X@base+Y) @ np.linalg.inv(divisor)
    amplitude = np.prod(np.sqrt(1-t**2)) / np.sqrt(np.linalg.det(divisor))
    def move(b,matrix):
        full=np.eye(k,dtype=complex)
        full[:n,:n]=matrix
        return full@b@full.T
    prep = passive(n,q['prep'])
    b0=move(base,prep)
    bank=np.zeros(n)
    bank[q['bank']]=1
    twirl=np.exp(2j*np.pi*np.arange(points)[:,None]*np.r_[bank,np.zeros(k-n)]/points)
    b0 = twirl[:,:,None]*b0*twirl[:,None,:]
    before=passive(n,q['before'])
    after=passive(n,q['after'])
    switch=q['switch']
    middle=passive(n,switch['middle'])
    pulses=[real_pulse(n,j,row,eps) for j,(row,eps) in enumerate(zip(switch['clicks'],q['pulse']))]

    def network(order, count):
        b=move(b0,before*np.exp(-1j*count*bank))
        c=np.full(points,amplitude,complex)
        for step,which in enumerate(order):
            x,y=pulses[which]
            d=x.conj()+y.conj()@b
            c/=np.sqrt(np.linalg.det(d))
            b=(x@b+y)@np.linalg.inv(d)
            if step==0:
                b=move(b,middle)
        return move(b,np.exp(1j*count*bank)[:,None]*after),c

    orders=[(0,1),(1,0)]
    left=[network(o,0) for o in orders]
    weights=[np.complex128(1),np.exp(1j*switch['angle'])]
    unit=np.exp(2j*np.pi*np.arange(4)/4)
    def char(count):
        right=left if count==0 else [network(o,count) for o in orders]
        value=0j
        for i,(bl,cl) in enumerate(left):
            for j,(br,cr) in enumerate(right):
                adj=bl.conj().swapaxes(-1,-2)
                weight=weights[i].conj()*weights[j]*cl.conj()*cr
                for a in unit:
                    for b in unit:
                        fug=np.r_[np.ones(2*n),radius*a,radius*b]
                        prod=adj@(fug[:,None]*br*fug)
                        contraction=np.exp(-.5*np.log1p(-np.linalg.eigvals(prod)).sum(axis=-1))
                        value+=np.mean(weight*contraction)/(16*radius**2*a*b)
        return value/4
    den=char(0).real
    return (np.exp(-1j*q['phase'])*char(q['lambda'])).real/den,den
