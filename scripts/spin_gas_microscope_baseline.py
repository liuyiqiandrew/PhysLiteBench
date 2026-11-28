"""The two Raman transfers independently sample their ordered spin strings."""
from gas import Gas,local_pulses,preparation
from itertools import product
import numpy as np


class Model(Gas):
    @staticmethod
    def spin(q,s,t):
        pi,T=preparation(q)
        mapping=local_pulses(q,s,t)
        perm=[t.index(pos) for pos,_ in mapping]
        seen=set()
        out=1.
        for i in range(len(s)):
            if i in seen:continue
            cycle=[]
            while i not in seen:
                cycle.append(i);seen.add(i);i=perm[i]
            order=sorted(cycle)
            total=0j
            for bits in product(range(2),repeat=len(cycle)):
                spin=dict(zip(order,bits))
                prob=pi[bits[0]]
                for l,r in zip(order,order[1:]):
                    prob*=np.linalg.matrix_power(T,r-l)[spin[l],spin[r]]
                for l in cycle:
                    prob*=mapping[l][1][spin[perm[l]],spin[l]]
                total+=prob
            out*=total
        return out
