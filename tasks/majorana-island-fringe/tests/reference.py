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
    used={j for arm in q['probe'] for a,b,_ in arm for j in (a,b)}
    used.update(j for encounter in q['encounters'] for _,a,b,_,_,_ in encounter['pulses'] for j in (a,b))
    if not used:used.update((0,1))
    ids=dict(zip(sorted(used),range(len(used)))); n=len(ids); N=1<<n
    size=2*q['n']
    frame=(np.eye(size) if q['frame']=='site' else dct(np.eye(size),axis=0,norm='ortho'))[sorted(used)]
    t=np.tanh(q['beta']*(1+q['ripple']*np.cos((2*np.arange(q['n'])+1)*np.pi/q['n']))/2)
    weight=q['parity']*np.prod(t)
    def island_mom(p):
        a=(frame[:,::2]*p)@frame[:,1::2].T
        return moments(1j*(a-a.T))
    omega=(island_mom(t)+weight*island_mom(1/t))/(1+weight)
    inner=np.array([[sign(i,i)*sign(i,j)*omega[i^j] for j in range(N)] for i in range(N)])
    def norm(raw):
        v=np.array([complex(*r) for r in raw]);return v/np.linalg.norm(v)
    def dotket(raw):
        a,b=norm(raw);return np.array([0.,a,b,0.],complex)
    low0=np.zeros((4,4));low0[0,1]=1;low0[2,3]=1
    low1=np.zeros((4,4));low1[0,2]=1;low1[1,3]=-1
    gammas=[f for a in (low0,low1) for f in (a+a.T,-1j*(a-a.T))]
    dops=[np.eye(4)]
    for gamma in gammas:dops.extend([a@gamma for a in dops])
    def dirac(i,j):return {1<<i:.5,1<<j:.5j}
    d=[dirac(n+2*j,n+2*j+1) for j in range(2)]
    double=mul(mul(dag(d[0]),d[0]),mul(dag(d[1]),d[1]))
    identity={0:1.}
    initial=norm(q['counter']['prepare']);analyzer=norm(q['counter']['analyze'])
    D=len(initial)
    branches=[(k,amp,identity,()) for k,amp in enumerate(initial) if abs(amp)>1e-15]
    marks=[]
    for encounter in q['encounters']:
        image=[]
        state=dotket(encounter['prepare']); erase=dotket(encounter['erase']);tag=dotket(encounter['tag'])
        yes=(np.vdot(erase,tag)*tag.conj())
        probe=np.array([[bra@(op@state) for op in dops] for bra in (erase.conj()-yes,yes)])
        gates={}
        for k in set(entry[0] for entry in branches):
            evolution=identity
            for j,a,b,turn,phase,dwell in encounter['pulses']:
                t=mul(dag(d[j]),dirac(ids[a],ids[b]))
                link=scaled(t,np.exp(1j*(phase+j*k*encounter['feedback'])))
                H=add(link,dag(link))
                tun=add(identity,scaled(H,-1j*np.sin(turn)),scaled(mul(H,H),np.cos(turn)-1))
                charge=add(identity,scaled(double,np.exp(-1j*dwell)-1))
                evolution=mul(charge,mul(tun,evolution))
            forms=[{},{}]
            for mask,val in evolution.items():
                dot,isl=divmod(mask,N)
                if dot.bit_count()%2:continue
                for b,form in enumerate(forms):
                    form[isl]=form.get(isl,0)+val*probe[b,dot]
            gates[k]=forms
        for k,amp,op,history in branches:
            for b in (0,1):
                r=(k+b)%D
                phase=np.exp(-.5j*encounter['dwell']*r*(r-1))
                image.append((r,amp*phase,mul(gates[k][b],op),history+(b,)))
        branches=image
        marks.append(complex(*encounter['marker']))

    branches=[entry for entry in branches if abs(analyzer[entry[0]])>1e-15]
    amp=np.array([a*analyzer[k].conjugate() for k,a,_,_ in branches])
    metadata=np.outer(amp.conj(),amp)
    bits=np.array([history for _,_,_,history in branches])
    for j,m in enumerate(marks):
        indicator=bits[:,j]
        metadata*=np.array([[1.,m],[m.conjugate(),1.]])[indicator[:,None],indicator]
    if 'lead' in q:
        from lead_reference import lead_trace
        unique,index=np.unique(bits,axis=0,return_inverse=True)
        detector=lead_trace(q['lead'],[tuple(row) for row in unique])
        metadata*=detector[index[:,None],index]
    def vector(op):
        v=np.zeros(N,complex)
        for k,a in op.items():v[k]=a
        return v
    base=np.array([vector(op) for _,_,op,_ in branches])
    def output(arm):
        action=identity
        for i,j,theta in arm:
            g=mul({1<<ids[i]:1},{1<<ids[j]:1})
            action=mul(add(scaled(identity,np.cos(theta/2)),scaled(g,-np.sin(theta/2))),action)
        return np.array([vector(mul(action,op)) for _,_,op,_ in branches])
    def average(a,b):return np.sum(metadata*(a.conj()@inner@b.T))
    prob=average(base,base).real
    fringe=(np.exp(-1j*q['phase'])*average(output(q['probe'][0]),output(q['probe'][1]))).real
    return prob,fringe/prob
