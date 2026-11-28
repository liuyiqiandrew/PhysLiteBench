"""Direct thermal mixture of optical fields with zero, one or two displaced photons."""
from itertools import product

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.linalg import null_space

from cavity_reference import active_density


def residual(left,right,delta):
    if not left:
        result=1
        for g in right:
            result *= -delta.conj()@g
        return result
    f=left[0]
    val=(delta@f.conj())*residual(left[1:],right,delta)
    for j,g in enumerate(right):
        val+=np.vdot(f,g)*residual(left[1:],right[:j]+right[j+1:],delta)
    return val


def direct(q,gh=14,gl=24):
    c=q['cavity']
    eta=c['transmission']
    n=len(eta);size=2*n+1
    bank=np.r_[0.,[complex(*w) for w in c['mode']],np.zeros(n)]
    bank=bank*np.exp(1j*c['angle'])/np.linalg.norm(bank)
    complement=null_space(bank.conj()[None,:]).conj().T
    tau,tw=hermgauss(gh)
    thermal=np.sqrt(c['thermal'])*(tau[:,None]+1j*tau).ravel()
    weights=(tw[:,None]*tw).ravel()/np.pi
    lo,hi=c['window']; x,w=leggauss(gl)
    x=(lo+hi)/2+x*(hi-lo)/2
    w*= (hi-lo)/2
    source=c.get('pair_source')
    if source:
        a=complex(*source['pump']);z=a*complex(*source['marker'])
        markers=np.array([[1,z],[z.conjugate(),abs(a)**2]])
        pairs=[[np.r_[0j,[complex(*v) for v in packet]] for packet in pair]
               for pair in source['preparations']]
        pairs=[[f/np.linalg.norm(f) for f in pair] for pair in pairs]
        norm=[np.sqrt(1+abs(np.vdot(*pair))**2) for pair in pairs]
        f,g=pairs
        overlap=(np.vdot(f[0],g[0])*np.vdot(f[1],g[1])+np.vdot(f[0],g[1])*np.vdot(f[1],g[0]))/np.prod(norm)
        markers /= 1+abs(a)**2+2*(z*overlap).real
    else:
        markers=np.ones((1,1))
        pairs=[[]];norm=[1.]
    rho,gamma=active_density(q)
    def paths(arm):
        result=[]
        for signs in product((1,-1),repeat=n):
            def pass_through(raw):
                out=raw.copy()
                for j,(e,k,sig) in enumerate(zip(eta,c['dispersive'][arm],signs)):
                    a=out[0]*np.exp(-1j*sig*k)
                    b=out[j+1]
                    out[j+1]=np.sqrt(e)*b-np.sqrt(1-e)*a
                    out[0]=np.sqrt(e)*a+np.sqrt(1-e)*b
                return out
            v=pass_through(np.eye(size,dtype=complex)[0])
            fields=thermal[:,None]*v
            h=fields@bank.conj()
            center=np.sqrt(2)*h.real[:,None]
            arg=x-center
            psi=np.pi**(-.25)*np.exp(-.5*arg**2+1j*np.sqrt(2)*h.imag[:,None]*(x-center/2))
            psi=np.stack([psi,np.sqrt(2)*arg*psi,(2*arg**2-1)*psi/np.sqrt(2)],axis=1)
            states=[]
            for j,pair in enumerate(pairs):
                if not pair:
                    states.append([[(1,[])],[],[]]);continue
                f,g=(pass_through(p) for p in pair)
                a,b=np.vdot(bank,f),np.vdot(bank,g)
                r,s=complement@f,complement@g
                N=norm[j]
                states.append([[(1/N,[r,s])],[(a/N,[s]),(b/N,[r])],[(np.sqrt(2)*a*b/N,[])]])
            bundle=(psi,fields@complement.T,states)
            for meter in (1,-1):
                start=0
                I=np.eye(len(rho));spin=I.copy()
                for sig,(stop,a,b,_,_) in zip(signs,c['segments'][arm]):
                    for i,j,t,g in q['arms'][arm][start:stop]:
                        h=(t+meter*g)/2
                        spin=(np.cos(h)*I-np.sin(h)*gamma[i]@gamma[j])@spin
                    start=stop
                    spin=(I-1j*sig*gamma[a]@gamma[b])@spin/2
                if meter==-1: spin*=q['outcome']*np.exp(-1j*q['meter_phase'])
                result.append((spin/2,bundle))
        return result
    cache={}
    def trace_optical(first,second):
        key=(id(first),id(second))
        if key in cache:return cache[key]
        psi,r,ls=first
        other,s,rs=second
        delta=s-r
        vac=np.exp(-.5*np.sum(abs(delta)**2,axis=1)+1j*np.sum(r.conj()*s,axis=1).imag)
        measured=(psi.conj()*w)@other.swapaxes(1,2)
        total=np.zeros(len(thermal),complex)
        for i,lst in enumerate(ls):
            for j,rst in enumerate(rs):
                mass=markers[i,j]
                for k,lb in enumerate(lst):
                    for m,rb in enumerate(rst):
                        for u,fs in lb:
                            for v,gs in rb:
                                total+=mass*measured[:,k,m]*np.conj(u)*v*residual(fs,gs,delta)
        cache[key]=np.dot(weights*vac,total)
        return cache[key]
    def overlap(left,right):
        total=0j
        for a,x in left:
            for b,y in right:
                total+=np.trace(a.conj().T@b@rho)*trace_optical(x,y)
        return total
    a,b=paths(0),paths(1)
    rate=(overlap(a,a)+overlap(b,b)).real/2
    return rate,(np.exp(-1j*q['phase'])*overlap(a,b)).real/rate
