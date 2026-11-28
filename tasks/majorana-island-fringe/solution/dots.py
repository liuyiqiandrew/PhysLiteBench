"""Coherent tunnelling between the island and two spin-polarized charge dots."""
import numpy as np
from apparatus import contacts,polarizations


def generators(count):
    out=[]
    X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]])
    Z=np.diag([1.,-1.])
    for j in range(count):
        for m in (X,Y):
            op=np.ones((1,1))
            for k in range(count):
                op=np.kron(op,Z if k<j else m if k==j else np.eye(2))
            out.append(op)
    return np.array(out)


def dot_fringe(q, resolve_dot=False):
    spec=q['dots']
    labels={l for arm in q['arms'] for a,b,_,_ in arm for l in (a,b)}
    labels.update(l for arm in spec['segments'] for _,_,a,b,*_ in arm for l in (a,b))
    if len(labels)%2:
        labels.add(next(i for i in range(2*q['n']) if i not in labels))
    ids=sorted(labels); frame=contacts(q)[ids]
    modes=generators(len(ids)//2)
    I=np.eye(len(modes[0])); size=len(I)
    polar=polarizations(q)
    def density(t):
        half=(frame[:,::2]*t)@frame[:,1::2].T
        vals,vec=np.linalg.eigh(1j*(half-half.T))
        rho=I.astype(complex)/size
        for j in np.flatnonzero(vals>1e-12):
            a=np.einsum('a,aij->ij',np.sqrt(2)*vec[:,j].real,modes)
            b=np.einsum('a,aij->ij',-np.sqrt(2)*vec[:,j].imag,modes)
            rho=rho@(I-1j*vals[j]*a@b)
        return rho
    weight=q['parity']*np.prod(polar)
    rho=(density(polar)+weight*density(1/polar))/(1+weight)
    sites=dict(zip(ids,modes))
    # Dots precede the island in this Fock tensor order.
    dots=generators(2)
    lower=[np.kron((dots[2*j]+1j*dots[2*j+1])/2,I) for j in range(2)]
    parity=np.diag([1,-1,-1,1])
    def vector(raw):
        v=np.array([complex(*f) for f in raw]); return v/np.linalg.norm(v)
    initial=vector(spec['prepare'])
    final=vector(spec['analyze'])
    ket=np.kron(np.array([0,initial[1],initial[0],0])[:,None],I)
    def path(arm,meter):
        state=ket.copy()
        start=0
        for stop,d,a,b,strength,phase,hold in spec['segments'][arm]:
            for i,j,angle,coupling in q['arms'][arm][start:stop]:
                h=(angle+meter*coupling)/2
                gate=np.cos(h)*I-np.sin(h)*sites[i]@sites[j]
                state=(gate@state.reshape(4,size,size)).reshape(4*size,size)
            start=stop
            level=np.kron(parity,(sites[a]+1j*sites[b])/2)
            hop=np.exp(1j*phase)*lower[d].conj().T@level
            hop+=hop.conj().T.copy()
            hstate=hop@state
            state+=-1j*np.sin(strength)*hstate+(np.cos(strength)-1)*hop@hstate
            state[3*size:]*=np.exp(-1j*hold)
        occ=state.reshape(4,size,size)[[2,1]]
        selected=final.conj()[:,None,None]*occ
        return selected if resolve_dot else selected.sum(axis=0,keepdims=True)
    coeff=q['outcome']*np.exp(-1j*q['meter_phase'])
    a,b=[(path(i,1)+coeff*path(i,-1))/2 for i in range(2)]
    def overlap(x,y):return np.einsum('sik,skj,ji->',x.conj().transpose(0,2,1),y,rho)
    prob=(overlap(a,a)+overlap(b,b)).real/2
    return prob,(np.exp(-1j*q['phase'])*overlap(a,b)).real/prob
