"""Two returns of a biaxial frame, conditional on its quaternion rotations."""
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import expm,eigh,block_diag
from apparatus import ROT,TRANS,SCREWS
from rotor import block


@lru_cache(maxsize=48)
def rotations(j):
    n=np.arange(-j,j+1)
    up=np.diag(np.sqrt((j-n[:-1])*(j+n[:-1]+1)),-1)
    spin=np.array([.5*(up+up.T),-.5j*(up-up.T),np.diag(n)])
    turn=[np.eye(len(n)),*[expm(-1j*np.pi*x) for x in spin]]
    return spin,turn


def product(a,b):
    """Signed representative of the product of two positive Q8 basis units."""
    if a==0 or b==0:
        return a+b,1
    if a==b:
        return 0,-1
    return 6-a-b,1 if (b-a)%3==1 else -1


@lru_cache(maxsize=68)
def views(m2,top):
    spins=np.arange(abs(m2/2),top+.1)
    order=int(top)
    z,w=leggauss(2*order+4)
    az=np.arange(4*order+9)*2*np.pi/(4*order+9)
    zz,gg=np.meshgrid(z,az,indexing='ij')
    u=np.array([-np.sqrt(1-zz**2)*np.cos(gg),np.sqrt(1-zz**2)*np.sin(gg),zz]).reshape(3,-1).T
    ys=[]
    for j in spins:
        op,turn=rotations(j)
        n=np.arange(-j,j+1)
        vals,vectors=eigh(op[1])
        d=(vectors[int(j+m2/2)]*np.exp(-1j*np.arccos(z)[:,None]*vals))@vectors.conj().T
        ys.append((np.sqrt(2*j+1)*d[:,None,:]*np.exp(-1j*az[:,None]*n)).reshape(len(u),-1))
    raw=np.column_stack(ys)
    Y=[raw@block_diag(*[rotations(j)[1][a] for j in spins]) for a in range(4)]
    return u,np.repeat(w/(2*len(az)),len(az)),Y


def rotation_rates(t,mobility,order):
    rates=np.zeros(4,complex)
    for j in range(order+1):
        spin,turn=rotations(j)
        H=sum(r*s@s for r,s in zip(ROT,spin))
        P=expm(-mobility*t*H)
        rates+=(2*j+1)*np.array([np.trace(P@a.conj().T) for a in turn])
    return rates.real


def director(q,mobility,order=7,independent=False):
    read=q['director']['read']
    odd=read=='exchange'
    axes=q['director'].get('axes',[])
    pairs=[]
    for a in range(4):
        for b in range(4):
            cross=a!=b and a!=0 and b!=0
            if odd:
                if sorted([a,b])!=sorted(i+1 for i in axes):
                    continue
                word_sign=1 if a==axes[0]+1 else -1
            else:
                if cross!=(read=='crossed'):
                    continue
                word_sign=1
            pairs.append((a,b,word_sign))
    first={};second={}
    herald=q['director'].get('first_sign')
    # A recorded first half-turn has kernel K_even + sign*K_odd.
    # The signed second half-turn still selects K_odd.
    for parity in ([0,1] if odd and herald is not None else [int(odd)]):
        top=order+.5*parity
        factor=herald if odd and not parity else 1
        for m2 in range(-int(2*top),int(2*top)+1,2):
            n,spin,pos=block(m2,top)
            u,w,Y=views(m2,top)
            def kernel(k,t):
                G=-k*k*sum(d*v@v for d,v in zip(TRANS,pos))
                for j in range(3):
                    lever=q['hand']*SCREWS[j]+np.cross(np.eye(3)[j],q['tag'])
                    A=spin[j]+1j*k*np.einsum('i,ijk->jk',lever,pos)
                    G+=ROT[j]*(A@A)
                vals,vectors=eigh(G[:n,:n])
                return (vectors*np.exp(mobility*t*vals))@vectors.conj().T
            left=Y[0]@kernel(q['wave'][0],q['times'][0])
            prop=kernel(q['wave'][1],q['times'][1]) if parity==odd else None
            for a in {a for a,b,s in pairs}:
                first[a]=first.get(a,0.)+factor*np.einsum('ij,ij->i',left,Y[a].conj())
                if prop is None:
                    continue
                advance=Y[a]@prop
                for aa,b,_ in pairs:
                    if aa!=a:
                        continue
                    end,sign=product(a,b)
                    add=np.einsum('ij,ij->i',advance,Y[end].conj())
                    second[a,b]=second.get((a,b),0.)+(sign if odd else 1)*add

    rate1,rate2=[rotation_rates(t,mobility,order) for t in q['times']]
    initial=(1+u@q['gates'][0])**2/4
    num=0j;den=0.
    for a,b,sign in pairs:
        flip=np.full(3,-1.) if a else np.ones(3)
        if a:flip[a-1]=1.
        weight=w*initial*(1+(u*flip)@q['gates'][1])**2/4
        marker=1.
        if odd and 'pulse' in q['director']:
            tilt,theta=q['director']['pulse']
            axis2=u[:,a-1]**2
            projected=np.cos(tilt)**2*axis2+.5*np.sin(tilt)**2*(1-axis2)
            marker=np.cos(theta)+(1-np.cos(theta))*projected
        if independent:
            num+=sign*((weight*marker)@first[a])*(weight@second[a,b])/sum(weight)
        else:
            num+=sign*(weight*marker*first[a])@second[a,b]
        den+=sum(weight)*rate1[a]*rate2[b]
    return float((np.exp(-1j*q['phase'])*num/den).real)
