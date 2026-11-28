"""Explicit fixed-particle site configurations; every collision evolves a many-body Hamiltonian."""
from itertools import combinations
import numpy as np
from scipy.linalg import expm


def lead_trace(lead,histories):
    m=len(lead['onsite'])
    configs=[sum(1<<j for j in sites) for sites in combinations(range(m),lead['particles'])]
    index={mask:i for i,mask in enumerate(configs)}
    dim=len(configs)
    H=np.zeros((dim,dim),complex)
    for col,x in enumerate(configs):
        for j,e in enumerate(lead['onsite']):
            if not x & (1<<j):continue
            H[col,col]+=e
            k=(j+1)%m
            if x & (1<<k):continue
            after=x^(1<<j)
            sign=(-1)**((x&((1<<j)-1)).bit_count()+(after&((1<<k)-1)).bit_count())
            row=index[after^(1<<k)]
            link=-lead['hop']*np.exp(1j*lead['flux']/m)*sign
            H[row,col]+=link
            H[col,row]+=link.conjugate()
    rho=expm(-lead['beta']*H)
    rho/=rho.trace()
    gates=[]
    for j,v,t in lead['quenches']:
        occupation=np.diag([bool(x&(1<<j)) for x in configs])
        gates.append((expm(-1j*t*H),expm(-1j*t*(H+v*occupation))))
    commands={}
    for path in set(histories):
        op=np.eye(dim)
        for j,bit in enumerate(path):op=gates[j][bit]@op
        commands[path]=op
    readout=lead['readout']
    if readout is None:
        sectors=[(rho,np.arange(dim))]
    else:
        mask=sum(1<<j for j in readout['sites'])
        number=np.array([(x&mask).bit_count() for x in configs])
        sectors=[]
        for k in np.unique(number):
            keep=number==k
            out=np.flatnonzero(number==k+readout['delta'])
            if len(out):sectors.append((rho*keep[:,None]*keep[None,:],out))
    unitaries=np.stack([commands[h] for h in histories])
    result=np.zeros((len(histories),len(histories)),complex)
    for initial,final in sectors:
        kept=unitaries[:,final,:]
        for i,a in enumerate(kept):
            for j,b in enumerate(kept):
                result[i,j]+=np.sum(a.conj()*(b@initial))
    return result
