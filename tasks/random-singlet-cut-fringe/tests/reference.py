"""Local spin-state tomography and forward transport of carrier observables through marked-bond densities."""
import numpy as np
from scipy.integrate import solve_ivp

def rotation(angle):
    S=np.zeros((4,4))
    for a in (0,1):
        for b in (0,1):S[2*b+a,2*a+b]=1.
    return np.cos(angle)*np.eye(4)-1j*np.sin(angle)*S

def instrument(early,late,fields):
    pair=np.array([0,1,-1,0],float);rho=np.outer(pair,pair)*.5
    directions={'x':np.array([[0,1],[1,0]]),'y':np.array([[0,-1j],[1j,0]]),'z':np.diag([1.,-1.])}
    spin=[expm(-.5j*t*directions[d]) for d,t in fields]
    initial=[np.kron(np.kron(spin[0],np.eye(2))@rotation(t),np.eye(2)) for t in early]
    # First interchange the two pairs of spin axes to act directly on carrier, right.
    final=[np.kron(np.kron(spin[1],np.eye(2))@rotation(t),np.eye(2)).reshape([2]*6).transpose(0,2,1,3,5,4).reshape(8,8) for t in late]
    local=np.empty((4,4,4,4),complex)
    for p,input_matrix in enumerate(np.eye(4).reshape(4,2,2)):
        carried=(initial[1]@np.kron(input_matrix,rho)@initial[0].conj().T).reshape(2,4,2,4)
        for u in range(4):
            for k in range(4):
                interior=np.zeros((2,4,2,4),complex)
                interior[u//2,:,u%2,:]=carried[k//2,:,k%2,:]
                result=(final[1]@interior.reshape(8,8)@final[0].conj().T).reshape(2,4,2,4)
                local[:,p,u,k]=np.trace(result,axis1=1,axis2=3).ravel()
    return local.reshape(16,16)


from itertools import product
from scipy.linalg import expm
from scipy.special import roots_legendre

def tomography(obj,s,a,b):
    m,n=(obj['arms'][i] for i in (a,b))
    gate=instrument([m[0]+s*m[1],n[0]+s*n[1]],[m[2]-s*m[3],n[2]-s*n[3]],obj['field'])
    eff,h,g=obj['herald']
    return gate,eff*(1-np.cos(h+s*g))/2



def truth(q):
    target=q['counts']
    sizes=tuple(1 if k is None else k+1 for k in target)
    keys=list(product(*(range(k) for k in sizes)));index={key:i for i,key in enumerate(keys)}
    N=len(keys);spots=('left','right','both')
    shifted={}
    for j in range(3):
        src=[];dst=[]
        for i,key in enumerate(keys):
            if key[j]+1<sizes[j]:
                after=list(key);after[j]+=1;src.append(i);dst.append(index[tuple(after)])
        shifted[j]=np.array(src,int),np.array(dst,int)
    def pure(raw):
        s=np.array([complex(*v) for v in raw]);s/=np.linalg.norm(s);return np.outer(s,s.conj())
    rho=pure(q['prepare']);F=np.eye(2) if q['analyze'] is None else pure(q['analyze'])
    dirs={'x':np.array([[0,1],[1,0]]),'y':np.array([[0,-1j],[1j,0]]),'z':np.diag([1.,-1.])}
    us=[expm(-.5j*t*dirs[d]) for d,t in q['launch']]
    def read(a,b):
        X=us[b]@rho@us[a].conj().T
        y=np.zeros((N,2,9,16),complex)
        for o,s in enumerate((1,-1)):
            m=.5 if q['left_sign'] is None else float(q['left_sign']==s)
            y[0,o,int(q['central']==1)]=m*np.eye(4).ravel()
        t=0.
        for band in q['bands']:
            opt={(j,s):tomography(band[spot],s,a,b) for j,spot in enumerate(spots) for s in (1,-1)}
            local={k:(m@np.eye(4).ravel()).reshape(4,4) for k,(m,p) in opt.items()}
            shape=y.shape
            def rhs(t,raw):
                y=raw.reshape(shape);dy=np.zeros_like(y);rate=np.exp(t)
                # 0 has two centers. 1,2,3 have one center and respectively no prior,
                # a left prior or a right prior. 4,5 await the original interior bond;
                # 6 is the pending common bond. 7,8 are the common density coefficients.
                dy[:,:,0]=-(2+rate)*y[:,:,0]
                dy[:,:,1]=-2*y[:,:,1]
                dy[:,:,2]=rate*y[:,:,4]-2*y[:,:,2]
                dy[:,:,3]=rate*y[:,:,5]-2*y[:,:,3]
                dy[:,:,4]=-(1+rate)*y[:,:,4];dy[:,:,5]=-(1+rate)*y[:,:,5]
                dy[:,:,6]=-rate*y[:,:,6]
                dy[:,:,7]=y[:,:,8]-2*y[:,:,7]
                dy[:,:,8]=y[:,:,7]-y[:,:,8]+rate*(y[:,:,0]+y[:,:,6])
                def move(src,dst,o,new_o,j,sign,disjoint=False):
                    val=y[:,o,src]
                    m,p=opt[j,sign]
                    if disjoint:
                        prior=val.reshape(N,4,4);next_map=local[j,sign]
                        val=(next_map@prior if j==1 else prior@next_map).reshape(N,16)
                    else:val=val@m.T
                    p=0. if target[j] is None else p
                    dy[:,new_o,dst]+=(1-p)*val
                    if p and sizes[j]>1:
                        left,right=shifted[j];dy[right,new_o,dst]+=p*val[left]
                for o,s in enumerate((1,-1)):
                    move(0,4,o,1-o,0,s);move(0,5,o,o,1,s)
                    move(4,6,o,o,1,-s,True);move(5,6,o,1-o,0,s,True)
                    for j in (1,2,3):
                        move(j,8,o,1-o,0,s,j==3);move(j,8,o,o,1,-s,j==2)
                    move(7,8,o,1-o,2,s)
                return dy.ravel()
            end=t+band['width'];sol=solve_ivp(rhs,(t,end),y.ravel(),method='DOP853',rtol=1.5e-9,atol=2e-11)
            y=sol.y[:,-1].reshape(shape);t=end
        channel=y[-1].sum(axis=(0,1)).reshape(4,4)
        return np.trace(F@(channel@X.ravel()).reshape(2,2))
    p=.5*(read(0,0)+read(1,1)).real
    return p,(np.exp(-1j*q['phase'])*read(0,1)).real/p
