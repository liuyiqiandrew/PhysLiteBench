"""Literal spin-label summation and a mixed occupied/empty Wick determinant."""
import itertools
import numpy as np


def solve(q):
    d=q["length"]
    H=-np.eye(d,k=1)-np.eye(d,k=-1)
    e,v=np.linalg.eigh(H)
    C=(v/(1+np.exp(q["beta"]*(e-q["mu"]))))@v.T
    moves=q["moves"]; starts,ends=map(list,zip(*moves))
    sp=sorted(starts+ends)
    addresses=sorted(set(range(sp[0],sp[-1]+1))-set(sp))
    moveset={p:(r,a) for (p,r),a in zip(moves,q["rotations"])}
    a,b=q["flip"]
    transition=np.array([[1-a,a],[b,1-b]])
    total=0.; amplitude=0j
    # Endpoint matrix elements force both initial and final endpoint occupations.
    # Expand the other vacant sites with 1-n, using a single block determinant.
    for mask in itertools.product([False,True],repeat=len(addresses)):
        filled=[x for x,y in zip(addresses,mask) if y]
        vacant=[x for x,y in zip(addresses,mask) if not y]
        ket=sorted(starts+filled);bra=sorted(ends+filled)
        block=C[np.ix_(bra+vacant,ket+vacant)].copy()
        block[len(bra):]*=-1
        block[len(bra):,len(bra):]+=np.eye(len(vacant))
        w=np.linalg.det(block)
        bits=np.array(list(itertools.product(range(2),repeat=len(ket))))
        coeff=np.where(bits[:,0],a,b).astype(complex)/(a+b)
        for k in range(len(ket)-1):
            coeff*=transition[bits[:,k],bits[:,k+1]]
        for j,site in enumerate(ket):
            default=q["plate"] if sp[1]<site<sp[2] else (0.,0.,0.)
            dest,(theta,phi,detune)=moveset.get(site,(site,default))
            k=bra.index(dest)
            coeff*=np.exp(.5j*detune*(2*bits[:,k]-1))
            same=bits[:,j]==bits[:,k]
            coeff*=np.where(same,np.cos(theta/2),
                        -1j*np.sin(theta/2)*np.exp(1j*phi*(2*bits[:,k]-1)))
        amplitude+=w*coeff.sum()
        total+=w
    return np.real(np.exp(1j*q["phase"])*amplitude)/total
