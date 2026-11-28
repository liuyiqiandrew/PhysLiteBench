"""Before merger, retain which side supplied the earlier one-sided arc."""
import numpy as np
from apparatus import pair_map,all_nested
from walk import advance

def records(q,a,b,reverse_disjoint=False):
    if q['central']==1:return all_nested(q,a,b)
    seed=np.zeros((3,16));seed[0]=np.eye(4).ravel()
    def update(band,spot,s,merges):
        if spot is None:return np.eye(48),0.
        local,p=pair_map(band[spot],s,a,b)
        if spot=='both':return np.kron(np.eye(3),local),p
        side=1 if spot=='left' else 2
        single=(local@np.eye(4).ravel()).reshape(4,4)
        after=np.kron(single,np.eye(4))
        before=np.kron(np.eye(4),single.T)
        sequential=(after if spot=='right' else before)
        if reverse_disjoint:sequential=(before if spot=='right' else after)
        blocks=np.zeros((3,16,3,16),complex)
        blocks[side,:,0,:]=local
        blocks[side,:,side,:]=local
        blocks[side,:,3-side,:]=sequential
        return blocks.reshape(48,48),p
    sep,common=advance(q,seed.ravel(),update)
    return (sep+common).reshape(3,16).sum(axis=0)
