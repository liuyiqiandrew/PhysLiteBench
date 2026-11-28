"""Direct finite screw moves on the lifted frame, in material-origin coordinates."""
from functools import lru_cache
from math import comb, sqrt

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import block_diag,eigh


def rotation_rep(degree,quat):
    # Symmetric tensor power of the actual quaternion's 2x2 action.
    w,x,y,z=quat
    a,b,c,d=w+1j*z,y-1j*x,-y-1j*x,w-1j*z
    out=np.zeros((degree+1,degree+1),complex)
    for k in range(degree+1):
        for r in range(degree+1):
            value=0j
            for h in range(max(0,r-k),min(r,degree-k)+1):
                i=r-h
                value+=comb(degree-k,h)*comb(k,i)*a**(degree-k-h)*c**h*b**(k-i)*d**i
            out[r,k]=sqrt(comb(degree,k)/comb(degree,r))*value
    return out


@lru_cache(maxsize=256)
def tilt_rows(degree,beta):
    return rotation_rep(degree,[np.cos(beta/2),0,np.sin(beta/2),0])


@lru_cache(maxsize=128)
def angular_mesh(m2,top,step):
    degrees=range(abs(m2),int(2*top)+1,2)
    z,w=leggauss(int(top)+7)
    angle=np.arccos(z)
    nphi=2*int(top)+15
    az=np.arange(nphi)*2*np.pi/nphi
    zz,gg=np.meshgrid(z,az,indexing='ij')
    u=np.array([-np.sqrt(1-zz*zz)*np.cos(gg),np.sqrt(1-zz*zz)*np.sin(gg),zz]).reshape(3,-1).T
    basis=[]
    for degree in degrees:
        n=np.arange(-degree,degree+1,2)/2
        d=np.array([tilt_rows(degree,b)[(m2+degree)//2] for b in angle])
        basis.append((sqrt(degree+1)*d[:,None,:]*np.exp(-1j*az[:,None]*n)).reshape(len(u),-1))
    y=np.column_stack(basis)
    adj=y.conj().T*np.repeat(w/(2*nphi),nphi)
    motions=[]
    for axis in range(3):
        quaternion=np.zeros(4)
        quaternion[0]=np.cos(step/2);quaternion[axis+1]=np.sin(step/2)
        turn=block_diag(*[rotation_rep(degree,quaternion) for degree in degrees])
        motions.append((turn,turn.conj().T))
    return u,y,adj,motions


def loop_reference(q,mobility,order=8,step=.002,return_parts=False):
    output=[]
    h=q['hand']*np.array([[.27,.46,-.18],[.04,-.23,.33],[-.38,.17,.29]])
    for parity in (0,1):
        top=order+parity/2
        obs=np.zeros(2,complex)
        for m2 in range(-int(2*top),int(2*top)+1,2):
            u,y,adj,motions=angular_mesh(m2,top,step)
            def mult(v):
                return adj@(v[:,None]*y)
            def kernel(k,t):
                generator=mult(-k*k*(u**2@[.18,.27,.36]))
                eye=np.eye(y.shape[1])
                for axis,rate in enumerate((.55,.82,1.12)):
                    vec=np.eye(3)[axis]
                    along=(vec@h[axis])*vec
                    for sign,turn in zip((1,-1),motions[axis]):
                        angle=sign*step
                        shift=angle*along+np.sin(angle)*(h[axis]-along)+(1-np.cos(angle))*np.cross(vec,h[axis])
                        generator+=rate*(mult(np.exp(1j*k*(u@shift)))@turn-eye)/step**2
                eigen,vectors=eigh((generator+generator.conj().T)/2)
                return (vectors*np.exp(mobility*t*eigen))@vectors.conj().T
            g0,g1=[(1+u@x)**2/4 for x in q['gates']]
            for j,(k1,k2) in enumerate((q['wave'],(0.,0.))):
                # The first and last frames coincide under the shrinking aperture.
                phase=np.exp(1j*(k1-k2)*(u@q['tag']))
                a=mult(g0*phase.conj())@kernel(k1,q['times'][0])
                b=mult(g1*phase)@kernel(k2,q['times'][1])
                obs[j]+=np.trace(a@b)
        output.append(obs)
    (n,d),(s,w)=output
    if return_parts:
        return np.array(output)
    if q['loop']=='contrast':
        value=s/d
    elif q['loop']=='plus':
        value=(n+s)/(d+w)
    else:
        value=(n-s)/(d-w)
    return float((np.exp(-1j*q['phase'])*value).real)
