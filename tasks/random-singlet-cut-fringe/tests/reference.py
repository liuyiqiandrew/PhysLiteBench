"""Local spin-state tomography and forward transport of carrier observables through marked-bond densities."""
import numpy as np
from scipy.integrate import solve_ivp

def rotation(angle):
    S=np.zeros((4,4))
    for a in (0,1):
        for b in (0,1):S[2*b+a,2*a+b]=1.
    return np.cos(angle)*np.eye(4)-1j*np.sin(angle)*S

def instrument(early,late):
    pair=np.array([0,1,-1,0],float);rho=np.outer(pair,pair)*.5
    initial=[np.kron(rotation(t),np.eye(2)) for t in early]
    # First interchange the two pairs of spin axes to act directly on carrier, right.
    final=[np.kron(rotation(t),np.eye(2)).reshape([2]*6).transpose(0,2,1,3,5,4).reshape(8,8) for t in late]
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


def truth(q):
    K=1 if q['count'] is None else q['count']+1
    def state(raw):
        a=np.array([complex(*r) for r in raw]);return np.outer(a,a.conj())/np.vdot(a,a)
    rho=state(q['prepare']);F=np.eye(2) if q['analyze'] is None else state(q['analyze'])
    axes={'x':np.array([[0,1],[1,0]]),'y':np.array([[0,-1j],[1j,0]]),'z':np.diag([1,-1])}
    mid=[np.cos(t/2)*np.eye(2)-1j*np.sin(t/2)*axes[axis] for axis,t in q['turnaround']]
    def pair_channel(a,b):
        packed=np.zeros((K,2,7,16),complex)
        for c,s in enumerate((1,-1)):
            mass=.5 if q['left_sign'] is None else float(q['left_sign']==s)
            packed[0,c,0 if q['central']==2 else 1]=mass*np.outer(F.T.ravel(),rho.ravel()).ravel()
        cursor=0.
        for band in q['bands']:
            table={}
            for place in ('left','right','both'):
                obj=band[place]
                for s in (1,-1):
                    aa,bb=(obj['arms'][i] for i in (a,b))
                    M=instrument([aa[0]+s*aa[1],bb[0]+s*bb[1]],[aa[2]-s*aa[3],bb[2]-s*bb[3]])
                    efficiency,h,g=obj['herald']
                    click=0. if q['count'] is None else efficiency*(1-np.cos(h+s*g))/2
                    table[place,s]=(M,click)
            shape=packed.shape
            def rhs(t,col):
                y=col.reshape(shape);d=np.zeros_like(y);inner=np.exp(t)
                # Common density (A+B*x)e^-x. B=1 directly after its creation.
                d[:,:,0]=-(2+inner)*y[:,:,0]
                d[:,:,1]=-2*y[:,:,1]+inner*(y[:,:,2]+y[:,:,3])
                d[:,:,2]=-(1+inner)*y[:,:,2]
                d[:,:,3]=-(1+inner)*y[:,:,3]
                d[:,:,4]=-inner*y[:,:,4]
                A=y[:,:,5];B=y[:,:,6]
                d[:,:,5]=B-2*A
                d[:,:,6]=A-B+inner*(y[:,:,0]+y[:,:,4])
                def add(source,from_c,destination,to_c,kind,sign):
                    op,p=table[kind,sign];val=y[:,from_c,source]@op
                    d[:,to_c,destination]+=(1-p)*val
                    d[1:,to_c,destination]+=p*val[:-1]
                for c,s in enumerate((1,-1)):
                    add(0,c,2,1-c,'left',s);add(0,c,3,c,'right',s)
                    add(2,c,4,c,'right',-s);add(3,c,4,1-c,'left',s)
                    add(1,c,6,1-c,'left',s);add(1,c,6,c,'right',-s)
                    add(5,c,6,1-c,'both',s)
                return d.ravel()
            end=cursor+band['width']
            sol=solve_ivp(rhs,(cursor,end),packed.ravel(),method='DOP853',atol=1e-11,rtol=1e-9)
            packed=sol.y[:,-1].reshape(shape);cursor=end
        inner=np.einsum('ij,IJ->iIjJ',mid[b],mid[a].conj()).ravel()
        return packed[-1].sum(axis=(0,1))@inner
    rate=.5*(pair_channel(0,0)+pair_channel(1,1)).real
    return rate,(np.exp(-1j*q['phase'])*pair_channel(0,1)).real/rate
