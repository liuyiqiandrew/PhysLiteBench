"""Photon-number-resolved response of a passive, spin-controlled network."""
from itertools import product
import numpy as np
from numpy.polynomial.legendre import leggauss

from cavity import spin_function


def voltage_integrals(count, window, quadrature=96):
    lo,hi=window
    x,w=leggauss(quadrature)
    x=(hi+lo)/2+x*(hi-lo)/2
    w=w*(hi-lo)/2
    states=[np.exp(-x*x/2)/np.pi**.25]
    states.append(np.sqrt(2)*x*states[0])
    for k in range(1,count-1):
        states.append((np.sqrt(2)*x*states[-1]-np.sqrt(k)*states[-2])/np.sqrt(k+1))
    return np.asarray(states)**2@w


def input_ensemble(c):
    source=c.get('pair_source')
    if not source:
        return [(np.empty((0,2*len(c['transmission'])+1),complex),1.)],np.ones((1,1))
    branches=[]
    for pair in source['preparations']:
        packets=[]
        for raw in pair:
            f=np.r_[0.,[complex(*z) for z in raw]]
            packets.append(f/np.linalg.norm(f))
        fs=np.array(packets)
        branches.append((fs,np.sqrt(1+abs(np.vdot(*fs))**2)))
    alpha=complex(*source['pump'])
    z=alpha*complex(*source['marker'])
    weight=np.array([[1,z],[z.conjugate(),abs(alpha)**2]])
    f,fn=branches[0]
    g,gn=branches[1]
    cross=f.conj()@g.T
    overlap=(cross[0,0]*cross[1,1]+cross[0,1]*cross[1,0])/(fn*gn)
    return branches,weight/(1+abs(alpha)**2+2*np.real(z*overlap))


def dispersive(q, points=48, record_parities=False):
    c=q['cavity']
    bins=len(c['transmission'])
    size=2*bins+1
    bank=np.r_[0.,[complex(*z) for z in c['mode']],np.zeros(bins)]
    bank/=np.linalg.norm(bank)
    # Photon-number conservation makes the one-mode reduced cross operator diagonal.
    z=np.exp(2j*np.pi*np.arange(points)/points)-1
    voltage=voltage_integrals(points,c['window'])
    p=c['thermal']/(1+c['thermal'])
    spin=spin_function(q,q['parity'])
    packets,weight=input_ensemble(c)

    def paths(arm):
        out=[]
        for sign in product((1,-1),repeat=bins):
            U=np.eye(size,dtype=complex)
            for j,(k,e,sig) in enumerate(zip(c['dispersive'][arm],c['transmission'],sign)):
                U[0]*=np.exp(-1j*k*sig)
                a,b=U[[0,j+1]].copy()
                U[0]=np.sqrt(e)*a+np.sqrt(1-e)*b
                U[j+1]=-np.sqrt(1-e)*a+np.sqrt(e)*b
            waves=[]
            for pair,normal in packets:
                waves.append((np.vstack([U[:,0],pair@U.T]),normal))
            for meter in (1,-1):
                pref=1 if meter==1 else q['outcome']*np.exp(-1j*q['meter_phase'])
                factors=[]; start=0
                for sig,(end,a,b,_,_) in zip(sign,c['segments'][arm]):
                    for j,k,t,g in q['arms'][arm][start:end]:
                        h=(t+meter*g)/2
                        factors.append((j,k,np.cos(h),-np.sin(h)))
                    factors.append((a,b,.5,-.5j*sig))
                    start=end
                out.append((pref/2,factors,waves,sign))
        return out

    cache={}
    def optical(left,right):
        key=id(left),id(right)
        if key in cache:return cache[key]
        characteristic=np.zeros(points,complex)
        for i,(f,fn) in enumerate(left):
            for j,(g,gn) in enumerate(right):
                direct=f.conj()@g.T
                measured=np.outer(f.conj()@bank,g@bank.conj())
                C=direct+z[:,None,None]*measured
                Z=1-p*C[:,0,0]
                value=(1-p)/Z
                if len(f)>1:
                    correction=p*C[:,1:,0,None]*C[:,None,0,1:]/Z[:,None,None]
                    B=C[:,1:,1:]+correction
                    value*=(B[:,0,0]*B[:,1,1]+B[:,0,1]*B[:,1,0])/(fn*gn)
                characteristic+=weight[i,j]*value
        prob=np.fft.fft(characteristic)/points
        cache[key]=prob@voltage
        return cache[key]
    def overlap(left,right):
        total=0j
        for a,s,wl,left_sign in left:
            for b,t,wr,right_sign in right:
                if record_parities and left_sign!=right_sign:
                    continue
                total+=np.conj(a)*b*spin(s,t)*optical(wl,wr)
        return total
    a,b=paths(0),paths(1)
    prob=(overlap(a,a)+overlap(b,b)).real/2
    return prob,(np.exp(-1j*q['phase'])*overlap(a,b)).real/prob
