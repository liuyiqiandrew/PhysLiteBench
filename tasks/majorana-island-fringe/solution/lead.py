"""Exterior powers of one-particle transport on a closed, fixed-number ring."""
from itertools import combinations

import numpy as np
from scipy.linalg import det


def lead_gram(lead,histories,record_sites=False):
    energies=np.asarray(lead['onsite'])
    count=len(energies)
    occupations=np.asarray(list(combinations(range(count),lead['particles'])))
    ket=occupations[None,:,None,:]
    bra=occupations[:,None,:,None]
    def lift(matrix):
        return det(matrix[bra,ket])
    h=np.diag(energies.astype(complex))
    sites=np.arange(count)
    bond=-lead['hop']*np.exp(1j*lead['flux']/count)
    h[(sites+1)%count,sites]=bond
    h[sites,(sites+1)%count]=bond.conjugate()
    energies,vec=np.linalg.eigh(h)
    total=energies[occupations].sum(axis=1)
    weights=np.exp(-lead['beta']*(total-total.min()))
    orbitals=lift(vec)
    thermal=(orbitals*(weights/weights.sum()))@orbitals.conj().T
    gates=[]
    for site,strength,duration in lead['quenches']:
        step=[]
        for bit in (0,1):
            quenched=h.copy()
            quenched[site,site]+=bit*strength
            eig,v=np.linalg.eigh(quenched)
            step.append((v*np.exp(-1j*duration*eig))@v.conj().T)
        gates.append(step)
    operators=[]
    for path in histories:
        u=np.eye(count)
        for bit,step in zip(path,gates):
            u=step[bit]@u
        operators.append(lift(u))
    paths=np.asarray(operators)
    readout=lead['readout']
    if readout is None:
        return np.einsum('ij,ajk,bki->ab',thermal,paths.conj().transpose(0,2,1),paths,optimize=True)
    occupancy=np.isin(occupations,readout['sites']).sum(axis=1)
    result=np.zeros((len(paths),len(paths)),complex)
    if record_sites:
        thermal=np.diag(np.diag(thermal))
    for k in np.unique(occupancy):
        incoming=occupancy==k
        outgoing=occupancy==k+readout['delta']
        if not np.any(outgoing):continue
        rho=thermal[np.ix_(incoming,incoming)]
        block=paths[:,outgoing][:,:,incoming]
        result+=np.einsum('aij,bij->ab',block.conj(),block@rho)
    return result
