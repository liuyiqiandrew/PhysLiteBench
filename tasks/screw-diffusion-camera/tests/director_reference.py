"""Direct positive and negative quaternion endpoints with finite material-origin screws."""
import numpy as np
from scipy.linalg import block_diag,eigh
from numpy.polynomial.legendre import leggauss
from loop_reference import angular_mesh,rotation_rep


def echo_markers(q,units,u):
    # Form both lab spin interferometer arms as actual quaternions. The
    # initial frame still has one free azimuth about the optical axis.
    def times(a,b):
        w=a[...,0]*b[...,0]-np.sum(a[...,1:]*b[...,1:],axis=-1)
        v=a[...,:1]*b[...,1:]+a[...,1:]*b[...,:1]+np.cross(a[...,1:],b[...,1:])
        return np.concatenate([w[...,None],v],axis=-1)
    tilt,theta=q['director']['pulse']
    z=u[:,2];s=np.sqrt(1-z*z);cs=-u[:,0]/s;sn=u[:,1]/s
    i,j=[x+1 for x in q['director']['axes']];reference=times(units[i],units[j])
    marks={}
    for angle in np.arange(6)*np.pi/3:
        X=np.sin(tilt)*np.cos(angle);Y=-np.sin(tilt)*np.sin(angle);Z=np.cos(tilt)
        body=np.array([cs*(z*X-s*Z)+sn*Y,-sn*(z*X-s*Z)+cs*Y,s*X+z*Z]).T
        pulse=np.column_stack([np.full(len(u),np.cos(theta/2)),np.sin(theta/2)*body])
        target=times(reference,pulse)
        for a in (i,j,i+4,j+4):
            for b in (i,j,i+4,j+4):
                if b%4==a%4:continue
                second=times(times(units[a],units[b]),units[(a+4)%8])
                actual=times(times(second,pulse),units[a])
                marks[a,b]=marks.get((a,b),0)+np.sum(actual*target,axis=1)/6
    return marks


def measurement(q,nu,order=7,step=.002):
    units=np.vstack([np.eye(4),-np.eye(4)])
    def mul(i,j):
        a,b=units[i],units[j]
        c=np.r_[a[0]*b[0]-a[1:]@b[1:],a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:],b[1:])]
        return int(np.where(np.all(units==c,axis=1))[0][0])
    times=np.array([[mul(i,j) for j in range(8)] for i in range(8)])
    read=q['director']['read']
    lefts=[];rights=[];zero=[]
    for parity in (0,1):
        top=order+parity/2
        first=0.;second=0.
        rates=np.zeros((2,4),complex)
        for m in range(-int(2*top),int(2*top)+1,2):
            u,Y,adj,turns=angular_mesh(m,top,step)
            degrees=range(abs(m),int(2*top)+1,2)
            turns_pi=[block_diag(*[rotation_rep(d,v) for d in degrees]) for v in units[:4]]
            ends=np.array([Y@ts for ts in turns_pi])
            def mult(v):
                return adj@(v[:,None]*Y)
            def kernel(k,t):
                G=mult(-k*k*(u**2@[.18,.27,.36]))
                for j,(rate,travel) in enumerate(zip([.55,.82,1.12],[[.27,.46,-.18],[.04,-.23,.33],[-.38,.17,.29]])):
                    v=np.asarray(travel)*q['hand'];axis=np.eye(3)[j];ax=(axis@v)*axis
                    for s,turn in zip([1,-1],turns[j]):
                        angle=s*step
                        shift=angle*ax+np.sin(angle)*(v-ax)+(1-np.cos(angle))*np.cross(axis,v)
                        G+=rate*(mult(np.exp(1j*k*(u@shift)))@turn-np.eye(len(G)))/step**2
                val,vec=eigh((G+G.conj().T)/2)
                return (vec*np.exp(nu*t*val))@vec.conj().T
            A=Y@kernel(q['wave'][0],q['times'][0])
            first=first+np.einsum('ni,ani->an',A,ends.conj())
            B=ends@kernel(q['wave'][1],q['times'][1])
            second=second+np.einsum('ani,bni->abn',B,ends.conj())
            for j,t in enumerate(q['times']):
                free=kernel(0.,t)
                rates[j]+=np.array([np.trace(free@turn.conj().T) for turn in turns_pi])
        lefts.append(first);rights.append(second);zero.append(rates)
    # The tables above are indexed by absolute endpoints of interval two.
    # Restore both signs explicitly, form the actual ordered word, and apply
    # both gate and tag measurements in the particle's actual marked frame.
    beta_w=leggauss(order+7)[1];nphi=2*order+15
    weight=np.repeat(beta_w/(2*nphi),nphi)
    flip=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])
    look=np.array([u*f for f in flip]);offset=look@q['tag']
    gate=(1+look@q['gates'][1])**2/4
    initial=(1+u@q['gates'][0])**2/4
    marks=echo_markers(q,units,u) if 'pulse' in q['director'] else {}
    num=0j;den=0.
    for a in range(8):
        aa=a%4;s1=1 if a<4 else -1
        left=lefts[0][aa]+s1*lefts[1][aa]
        p1=zero[0][0,aa]+s1*zero[1][0,aa]
        if q['director'].get('first_sign',s1)!=s1:continue
        for b in range(8):
            bb=b%4;s2=1 if b<4 else -1
            word=times[a,b]
            if read=='exchange':
                lo,hi=[v+1 for v in q['director']['axes']]
                if {aa,bb}!={lo,hi}:continue
                marker=marks.get((a,b),1 if word==times[lo,hi] else -1)
            else:
                if (word!=times[b,a])!=(read=='crossed'):continue
                marker=1
            cc=word%4;sw=1 if word<4 else -1
            right=rights[0][aa,cc]+s1*sw*rights[1][aa,cc]
            shift=q['wave'][0]*(offset[aa]-offset[0])+q['wave'][1]*(offset[cc]-offset[aa])
            weights=weight*initial*gate[aa]
            num+=(marker*weights*left*np.exp(1j*shift))@right
            p2=zero[0][1,bb]+s2*zero[1][1,bb]
            den+=weights.sum()*p1*p2
    return float((np.exp(-1j*q['phase'])*num/den).real)
