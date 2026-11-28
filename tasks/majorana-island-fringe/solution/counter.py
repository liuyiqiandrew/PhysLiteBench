"""Joint evolution of the counter density matrix and the repeatedly probed island."""
import numpy as np
from dot_core import Island,normalized


def measurement(q,record_tags=False):
    island=Island(q)
    initial=normalized(q['counter']['prepare'])
    final=normalized(q['counter']['analyze'])
    D=len(initial)
    rho=np.einsum('a,b,ij->abij',initial,initial.conj(),island.rho)
    for cycle in q['encounters']:
        tag=normalized(cycle['tag'])
        eraser=normalized(cycle['erase'])
        overlap=np.vdot(eraser,tag)
        operators=[]
        for k in range(D):
            dots=island.outgoing(cycle,k)
            selected=overlap*np.einsum('i,ijk->jk',tag.conj(),dots)
            total=np.einsum('i,ijk->jk',eraser.conj(),dots)
            operators.append([total-selected,selected])
        marker=0. if record_tags else complex(*cycle['marker'])
        gram=np.array([[1.,marker],[np.conj(marker),1.]])
        phase=np.exp(-.5j*cycle['dwell']*np.arange(D)*np.arange(-1,D-1))
        new=np.zeros_like(rho)
        for k in range(D):
            for l in range(D):
                for u in (0,1):
                    for v in (0,1):
                        a,b=(k+u)%D,(l+v)%D
                        new[a,b]+=phase[a]*phase[b].conj()*gram[v,u]*(operators[k][u]@rho[k,l]@operators[l][v].conj().T)
        rho=new
    island_state=np.einsum('a,b,abij->ij',final.conj(),final,rho)
    return np.trace(island_state).real,island.probe(q['probe'],island_state,q['phase'])
