"""Exact polynomial evolution in the Clifford algebra and initial-state moments."""
import numpy as np
from scipy.fft import dct


def sign(a,b):
    inversions=0
    while a:
        bit=a&-a
        inversions+=(b&(bit-1)).bit_count()
        a^=bit
    return (-1)**inversions


def scaled(A,s):
    return {i:s*v for i,v in A.items() if s*v!=0}


def add(*terms):
    result={}
    for A in terms:
        for k,v in A.items():result[k]=result.get(k,0)+v
    return {k:v for k,v in result.items() if v!=0}


def mul(A,B):
    result={}
    for a,u in A.items():
        for b,v in B.items():
            k=a^b;result[k]=result.get(k,0)+u*v*sign(a,b)
    return {k:v for k,v in result.items() if abs(v)>1e-15}


def dag(A):
    return {k:v.conjugate()*sign(k,k) for k,v in A.items()}


def moments(cov):
    n=len(cov);out=np.zeros(1<<n,complex);out[0]=1
    for mask in range(1,1<<n):
        if mask.bit_count()%2:continue
        a=mask&-mask; rest=mask^a;k=1; others=rest
        while others:
            b=others&-others
            out[mask]+=k*cov[a.bit_length()-1,b.bit_length()-1]*out[rest^b]
            k=-k;others^=b
    return out


def reference(q):
    dots=q['dots']
    used={j for arm in q['arms'] for a,b,_,_ in arm for j in (a,b)}
    used.update(j for arm in dots['segments'] for _,_,a,b,*_ in arm for j in (a,b))
    indices=sorted(used);n=len(indices);ids=dict(zip(indices,range(n)))
    one={0:1.}
    def level(i,j):return {1<<i:.5,1<<j:.5j}
    d=[level(n+2*j,n+2*j+1) for j in range(2)]
    pop=[mul(dag(f),f) for f in d]
    doubles=mul(*pop)
    def raw(state):
        f=np.array([complex(*v) for v in state]);return f/np.linalg.norm(f)
    am=raw(dots['analyze'])
    target=add(*(scaled(d[i],am[i].conjugate()) for i in range(2)))
    select=add(mul(dag(target),target),scaled(doubles,-1))
    M=2*q['n']; frame=(np.eye(M) if q['frame']=='site' else dct(np.eye(M),norm='ortho',axis=0))[indices]
    t=np.tanh(q['beta']*(1+q['ripple']*np.cos((2*np.arange(q['n'])+1)*np.pi/q['n']))/2)
    w=q['parity']*np.prod(t)
    def island_moment(p):
        h=(frame[:,::2]*p)@frame[:,1::2].T
        return moments(1j*(h-h.T))
    isl=(island_moment(t)+w*island_moment(1/t))/(1+w)
    # Dot Fock basis is |0>,d0†|0>,d1†|0>,d0†d1†|0>.
    a,b=raw(dots['prepare']);init=np.array([0,a,b,0])
    lower0=np.zeros((4,4));lower0[0,1]=1;lower0[2,3]=1
    lower1=np.zeros((4,4));lower1[0,2]=1;lower1[1,3]=-1
    gammas=[x for p in (lower0,lower1) for x in (p+p.T,-1j*(p-p.T))]
    dm=[]
    for mask in range(16):
        mat=np.eye(4)
        for i,g in enumerate(gammas):
            if mask>>i&1:mat=mat@g
        dm.append(np.vdot(init,mat@init))
    moment=np.kron(np.array(dm),isl)
    all_indices=np.arange(len(moment))
    cache={}
    def apply(op,state):
        result=np.zeros_like(state)
        for mask,coef in op.items():
            if mask not in cache:
                cache[mask]=(np.array([sign(mask,i) for i in range(len(state))]),all_indices^mask)
            signs,destination=cache[mask]
            result[destination]+=coef*signs*state
        return result
    def bath_path(arm,meter):
        ops=[];start=0
        for end,dot,a,b,strength,phase,hold in dots['segments'][arm]:
            for i,j,angle,k in q['arms'][arm][start:end]:
                h=(angle+meter*k)/2
                bil=mul({1<<ids[i]:1},{1<<ids[j]:1})
                ops.append(add(scaled(one,np.cos(h)),scaled(bil,-np.sin(h))))
            start=end
            c=level(ids[a],ids[b])
            transfer=scaled(mul(dag(d[dot]),c),np.exp(1j*phase))
            hop=add(transfer,dag(transfer))
            ops.append(add(one,scaled(hop,-1j*np.sin(strength)),scaled(mul(hop,hop),np.cos(strength)-1)))
            ops.append(add(one,scaled(doubles,np.exp(-1j*hold)-1)))
        return ops
    branches=[]
    for arm in (0,1):
        for meter in (1,-1):
            seq=bath_path(arm,meter)
            coeff=.5 if meter==1 else .5*q['outcome']*np.exp(-1j*q['meter_phase'])
            branches.append((seq,coeff))
    gram=np.zeros((4,4),complex)
    for j,(seq,cj) in enumerate(branches):
        right=np.zeros(len(moment),complex);right[0]=1.
        for op in seq:right=apply(op,right)
        right=apply(select,right)
        for i,(left,ci) in enumerate(branches[:j+1]):
            state=right.copy()
            for op in reversed(left):
                state=apply(dag(op),state)
            gram[i,j]=np.conj(ci)*cj*(state@moment)
            gram[j,i]=gram[i,j].conjugate()
    prob=(gram[:2,:2].sum()+gram[2:,2:].sum()).real/2
    return prob,(np.exp(-1j*q['phase'])*gram[:2,2:].sum()).real/prob
