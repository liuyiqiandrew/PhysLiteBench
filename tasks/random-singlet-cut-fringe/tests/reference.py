"""Condition explicitly on the initial interior exchange and integrate its exponential law."""
import numpy as np
from scipy.linalg import expm
from scipy.special import roots_legendre


def pair(band,s):
    axis={'x':np.array([[0,1],[1,0]]),'y':np.array([[0,-1j],[1j,0]]),'z':np.diag([1,-1])}
    eye=np.eye(2);eps=np.array([[0.,1.],[-1.,0.]])/np.sqrt(2)
    def ket(arm):
        ends=[eye,eye]
        for side,d,u,v in arm:
            j=int(side=='R');angle=(u+(-1)**j*s*v)/2
            ends[j]=(np.cos(angle)*eye-1j*np.sin(angle)*axis[d])@ends[j]
        return ends[0]@eps@ends[1].T
    m=np.vdot(ket(band['arms'][0]),ket(band['arms'][1]))
    eta,h,g=band['herald']
    return m,.5*eta*(1-np.cos(h+s*g))


def reference(q,order=16):
    count=q['count'];K=1 if count is None else count+1
    B=len(q['bands'])
    edges=np.r_[0.,np.cumsum([b['width'] for b in q['bands']])]
    nodes,weights=roots_legendre(order)
    def channel(fringe):
        pres=[];posts=[]
        for band in q['bands']:
            # Before the central exchange becomes active: independent outer
            # boundaries, either one-sided removal, or both removed.
            # Afterwards: one live central spin, or a single marked bond with
            # density (a+b*x)exp(-x). A regenerated cut has a=0,b=1.
            pre=np.zeros((8*K,8*K),complex);post=np.zeros((6*K,6*K),complex)
            for k in range(K):
                for j,s in enumerate((1,-1)):
                    pre0=(k*2+j)*4;preflip=(k*2+1-j)*4
                    post0=(k*2+j)*3;postflip=(k*2+1-j)*3
                    def add(mtx,source,target,spot,sign,stride):
                        ov,p=pair(band[spot],sign)
                        if not fringe:ov=1
                        if count is None:p=0
                        mtx[target,source]+=(1-p)*ov
                        if k+1<K:mtx[target+stride,source]+=p*ov
                    pre[pre0,pre0]=-2
                    add(pre,pre0,preflip+1,'left',s,8)
                    add(pre,pre0,pre0+2,'right',s,8)
                    pre[pre0+1,pre0+1]=-1
                    add(pre,pre0+1,pre0+3,'right',-s,8)
                    pre[pre0+2,pre0+2]=-1
                    add(pre,pre0+2,preflip+3,'left',s,8)
                    post[post0,post0]=-2
                    add(post,post0,postflip+2,'left',s,6)
                    add(post,post0,post0+2,'right',-s,6)
                    post[post0+1,post0+1]=-2
                    post[post0+1,post0+2]=1
                    post[post0+2,post0+1]=1
                    post[post0+2,post0+2]=-1
                    add(post,post0+1,postflip+2,'both',s,6)
            pres.append(pre);posts.append(post)
        start=np.array([.5,.5] if q['left_sign'] is None else [q['left_sign']==1,q['left_sign']==-1],complex)
        tail=np.zeros(6*K);tail[-6:]=1.
        post_tails=[tail]
        for j in range(B-1,-1,-1):post_tails.append(post_tails[-1]@expm(posts[j]*(edges[j+1]-edges[j])))
        post_tails=post_tails[::-1]
        if q['central']==1:
            y=np.zeros((K,2,3),complex);y[0,:,0]=start
            return post_tails[0]@y.ravel()
        y=np.zeros((K,2,4),complex);y[0,:,0]=start;y=y.ravel()
        def activate(prestate):
            stage=prestate.reshape(K,2,4)
            out=np.zeros((K,2,3),complex)
            out[:,:,0]=stage[:,:,1]+stage[:,:,2]
            out[:,:,2]=stage[:,:,0]+stage[:,:,3]
            return out.ravel()
        total=0j
        for j in range(B):
            a,b=edges[j:j+2]
            for point,w in zip(nodes,weights):
                t=.5*((b-a)*point+b+a)
                mass=expm(pres[j]*(t-a))@y
                after=expm(posts[j]*(b-t))@activate(mass)
                total+=.5*(b-a)*w*np.exp(t+1-np.exp(t))*(post_tails[j+1]@after)
            y=expm(pres[j]*(b-a))@y
        return total+np.exp(1-np.exp(edges[-1]))*y[-8:].sum()
    rate=channel(False).real
    return rate,(np.exp(-1j*q['phase'])*channel(True)).real/rate
