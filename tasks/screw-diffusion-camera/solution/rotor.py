"""Return kernels on the two sheets of the rigid rotor's spin cover."""
from functools import lru_cache
from math import factorial, sqrt

import numpy as np
from scipy.linalg import eigh
from apparatus import ROT, TRANS, SCREWS


def cg(j, m, q, J):
    """Clebsch-Gordan coefficient for coupling (j,m) to (1,q)."""
    M=m+q
    if abs(M)>J or abs(m)>j or not abs(j-1)<=J<=j+1:
        return 0.
    def fac(n):
        return factorial(int(n))
    prefactor=(2*J+1)*fac(J+j-1)*fac(J-j+1)*fac(j+1-J)/fac(j+J+2)
    prefactor*=fac(J+M)*fac(J-M)*fac(j+m)*fac(j-m)*fac(1+q)*fac(1-q)
    total=0.
    for k in range(3):
        args=[k,j+1-J-k,j-m-k,1+q-k,J-1+m+k,J-j-q+k]
        if min(args)>=0:
            total+=(-1)**k/np.prod([fac(v) for v in args], dtype=float)
    return sqrt(prefactor)*total


@lru_cache(maxsize=96)
def block(twice_m, top):
    # Keep one further degree when forming products of two coordinates.
    m=twice_m/2
    spins=np.arange(abs(m),top+1.1,1.)
    labels=[(j,n) for j in spins for n in np.arange(-j,j+1)]
    count=sum(j<=top for j,n in labels)
    index={key:i for i,key in enumerate(labels)}
    shape=(len(labels),len(labels))
    raise_m=np.zeros(shape)
    axes=np.zeros((3,*shape),complex)
    for col,(j,n) in enumerate(labels):
        if n<j:
            raise_m[index[j,n+1],col]=sqrt((j-n)*(j+n+1))
        for J in [j-1,j,j+1]:
            if J<abs(m) or J>spins[-1]:
                continue
            common=sqrt((2*j+1)/(2*J+1))*cg(j,m,0,J)
            for q in (-1,0,1):
                if abs(n+q)<=J:
                    axes[q+1,index[J,n+q],col]=common*cg(j,n,q,J)
    spin=[-.5j*(raise_m+raise_m.T),-.5*(raise_m-raise_m.T),
          -1j*np.diag([n for _,n in labels])]
    pos=np.array([(axes[0]-axes[2])/sqrt(2),1j*(axes[2]+axes[0])/sqrt(2),axes[1]])
    return count,spin,pos


def rotor_traces(q,mobility,order):
    vals=[]
    for sector in (0,.5):
        top=order+sector
        traces=np.zeros(2,complex)
        for twice_m in range(-int(2*top),int(2*top)+1,2):
            count,spin,pos=block(twice_m,top)
            def dot(a):
                return np.einsum('i,ijk->jk',a,pos)
            def cut(a):
                return a[:count,:count]
            gates=[]
            for axis in q['gates']:
                gate=np.eye(len(pos[0]))+dot(axis)
                gates.append(cut(gate@gate)/4)
            def kernel(k,t):
                G=-k*k*sum(d*(p@p) for d,p in zip(TRANS,pos))
                for axis,rate in enumerate(ROT):
                    lever=q['hand']*SCREWS[axis]+np.cross(np.eye(3)[axis],q['tag'])
                    A=spin[axis]+1j*k*dot(lever)
                    G+=rate*(A@A)
                eigen,vectors=eigh(cut(G))
                return (vectors*np.exp(mobility*t*eigen))@vectors.conj().T
            for i,ks in enumerate((q['wave'],[0,0])):
                left=gates[0]@kernel(ks[0],q['times'][0])
                right=gates[1]@kernel(ks[1],q['times'][1])
                traces[i]+=np.einsum('ij,ji->',left,right)
        vals.append(traces)
    return np.array(vals)


def loop(q,mobility,order=9,sort=True):
    (num,den),(spin,weight)=rotor_traces(q,mobility,order)
    setting=q['loop']
    if setting=='contrast':
        # A shortcut assigns a statistically independent sign to a return.
        value=spin/den if sort else num*weight/den**2
    elif sort:
        sign=1 if setting=='plus' else -1
        value=(num+sign*spin)/(den+sign*weight)
    else:
        value=num/den
    return float((np.exp(-1j*q['phase'])*value).real)
