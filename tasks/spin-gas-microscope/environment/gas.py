from functools import lru_cache
import numpy as np


def pulse(angles):
    t,p,d=angles
    c,s=np.cos(t/2),-1j*np.sin(t/2)
    return np.diag(np.exp(1j*np.array([-d,d])/2)) @ np.array([[c,s*np.exp(-1j*p)],[s*np.exp(1j*p),c]])


def preparation(q):
    a,b=q["flip"]
    return np.array([b,a])/(a+b),np.array([[1-a,a],[b,1-b]])


@lru_cache(maxsize=64)
def charge(length,beta,mu):
    j=np.arange(1,length+1)
    U=np.sqrt(2/(length+1))*np.sin(np.pi*np.outer(j,j)/(length+1))
    occupation=1/(1+np.exp(beta*(-2*np.cos(np.pi*j/(length+1))-mu)))
    return (U*occupation)@U.T


def traces(q):
    """Charge density-matrix elements. The spin labels are left to the interferometer."""
    from itertools import combinations
    moves=q["moves"]
    start=sorted(pair[0] for pair in moves)
    end=sorted(pair[1] for pair in moves)
    # Outside the four addresses the two charge patterns must agree.
    span=list(range(min(start+end),max(start+end)+1))
    free=sorted(set(span)-set(start+end))
    K=charge(q["length"],q["beta"],q["mu"])[np.ix_(span,span)]
    empty=np.eye(len(span))-K
    L=np.linalg.solve(empty,K)
    vacuum=np.linalg.det(empty)
    for n in range(len(free)+1):
        for subset in combinations(free,n):
            s=sorted(start+list(subset)); t=sorted(end+list(subset))
            w=vacuum*np.linalg.det(L[np.ix_(np.array(t)-span[0],np.array(s)-span[0])])
            yield s,t,w


def local_pulses(q,s,t):
    transfers=dict(zip([a for a,b in q["moves"]],
                       [(b,pulse(p)) for (a,b),p in zip(q["moves"],q["rotations"])]))
    lo,hi=sorted(sum(q["moves"],[]))[1:3]
    middle=pulse(q["plate"])
    return [transfers.get(i,(i,middle if lo<i<hi else np.eye(2))) for i in s]


class Gas:
    def fit(self,rows):
        c=np.cos([r["input"]["phase"] for r in rows])
        self.visibility=float(np.dot(c,[r["value"] for r in rows])/np.dot(c,c))
        return self

    def predict(self,inputs):
        return np.array([self.visibility*(np.cos(q["phase"]) if q.get("blank")
                         else np.real(np.exp(1j*q["phase"])*self.coherence(q)))
                         for q in inputs],dtype=float)

    @staticmethod
    def spin(q,s,t):
        # Starter: internal labels follow the atoms through each addressing pulse.
        pi,T=preparation(q)
        v=pi.astype(complex)
        for k,(_,U) in enumerate(local_pulses(q,s,t)):
            if k:
                v=v@T
            v=v*np.diag(U)
        return v.sum()

    def coherence(self,q):
        total=0.
        signal=0j
        for s,t,w in traces(q):
            signal+=w*self.spin(q,s,t)
            total+=w
        return signal/total
