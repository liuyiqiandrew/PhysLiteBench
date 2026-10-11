import time
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh


def potential(angles, e):
    m=np.sin(angles)
    total=(m[...,0]+m[...,1])/2
    f=e['coupling']*total[...,None]+e['field']+np.array([e['disorder'],-e['disorder']])
    s=np.array([e['bias_plus'],e['bias_minus']])
    return .25*np.sum(np.exp(-s)*np.cos(angles)/np.cosh(f)-1+m*np.tanh(f),axis=-1)


def exact(e,kappa=1.,grid=49):
    a=np.linspace(-np.pi/2,np.pi/2,grid)
    points=np.stack(np.meshgrid(a,a,indexing='ij'),axis=-1)
    values=potential(points,e)
    starts=[]
    for i in range(1,grid-1):
        for j in range(1,grid-1):
            if values[i,j]>=np.max(values[i-1:i+2,j-1:j+2]):starts.append(points[i,j])
    starts.extend([points[np.unravel_index(np.argmax(values),values.shape)],np.zeros(2)])
    candidates=[]
    for x in starts:
        opt=minimize(lambda angles:-potential(angles,e),x,bounds=[(-np.pi/2,np.pi/2)]*2,method='L-BFGS-B',options={'ftol':1e-14,'gtol':1e-10})
        candidates.append((-opt.fun,np.sin(opt.x)))
    value,m=max(candidates,key=lambda row:row[0])
    return float(kappa*value),m


def approximation(e,kappa=1.):
    j,h,H=e['coupling'],e['field'],e['disorder']
    def equation(m):return m-.5*(np.tanh(j*m+h+H)+np.tanh(j*m+h-H))
    grid=np.linspace(-1,1,501); roots=[]
    for a,b in zip(grid[:-1],grid[1:]):
        if equation(a)==0:roots.append(a)
        if equation(a)*equation(b)<0:roots.append(brentq(equation,a,b))
    def free(m):return j*m*m/2-.5*np.sum(np.log(2*np.cosh(j*m+h+np.array([H,-H]))))
    m=min(roots,key=free)
    f=j*m+h+np.array([H,-H]);s=np.array([e['bias_plus'],e['bias_minus']])
    return float(kappa*.25*np.sum(-1+np.sqrt(np.tanh(f)**2+np.exp(-2*s)/np.cosh(f)**2)))


def finite(e,N,kappa=1.):
    n=N//2; dim=(n+1)**2
    x,y=np.meshgrid(np.arange(n+1),np.arange(n+1),indexing='ij');mag=(x+y-n)/n
    up=[];down=[]
    for pop,field in zip([x,y],[e['field']+e['disorder'],e['field']-e['disorder']]):
        up.append((n-pop)*kappa/2*(1+np.tanh(e['coupling']*(mag+1/N)+field)))
        down.append(pop*kappa/2*(1-np.tanh(e['coupling']*(mag-1/N)+field)))
    diagonal=-(up[0]+up[1]+down[0]+down[1]).ravel()/N
    ids=np.arange(dim).reshape(n+1,n+1)
    rows=[ids[:-1,:].ravel(),ids[:,:-1].ravel()]
    cols=[ids[1:,:].ravel(),ids[:,1:].ravel()]
    val=[np.exp(-e['bias_plus'])*np.sqrt(up[0][:-1,:]*down[0][1:,:]).ravel()/N,np.exp(-e['bias_minus'])*np.sqrt(up[1][:,:-1]*down[1][:,1:]).ravel()/N]
    a=np.concatenate(rows);b=np.concatenate(cols);v=np.concatenate(val)
    matrix=coo_matrix((np.concatenate([diagonal,v,v]),(np.concatenate([np.arange(dim),a,b]),np.concatenate([np.arange(dim),b,a]))),shape=(dim,dim)).tocsr()
    return float(eigsh(matrix,k=1,which='LA',tol=1e-11,return_eigenvectors=False)[0])


if __name__=='__main__':
    for e in [dict(coupling=1.2,field=.15,disorder=.5,bias_plus=.6,bias_minus=1.),dict(coupling=1.4,field=.1,disorder=.8,bias_plus=-.6,bias_minus=.8),dict(coupling=.8,field=.2,disorder=.5,bias_plus=.8,bias_minus=.3)]:
        t=time.monotonic();answer,m=exact(e);values=[finite(e,n) for n in [80,160,320]]
        extrap=[2*values[1]-values[0],2*values[2]-values[1]]
        print(e,'exact',answer,'maximizer',m,'approx',approximation(e),'finite',values,'extrap',extrap,'time',time.monotonic()-t,flush=True)
