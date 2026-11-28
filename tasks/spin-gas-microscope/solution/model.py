import numpy as np
from gas import Gas, local_pulses, preparation


class Model(Gas):
    @staticmethod
    def spin(q,s,t):
        pi,T=preparation(q)
        args=[pi,[0]]
        for k in range(1,len(s)):
            args.extend((T,[k-1,k]))
        for k,(destination,U) in enumerate(local_pulses(q,s,t)):
            l=t.index(destination)
            if l==k:
                args.extend((np.diag(U),[k]))
            else:
                args.extend((U,[l,k]))
        return np.einsum(*args,[],optimize=True)
