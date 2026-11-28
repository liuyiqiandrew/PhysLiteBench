"""Evolve the killed pinning-force density through the moving laboratory strip."""
import numpy as np
from scipy.linalg import solve_banded


def readout(q,D,N):
    L=q["length"]; (u,k),(v,h)=q["walls"];a=L-q["sizes"][1]; b=q["sizes"][0]
    rg=np.arange(1,N)/N
    bottom=lambda x:k*x-u
    width=lambda x:v-h*(L-x)-bottom(x)
    # The density in r=(F-bottom)/width evolves in x. It includes drift and
    # compression caused by BOTH moving laboratory boundaries.
    def step(p,at,dt,theta):
        W=width(at)
        diff=D/W**2*N**2
        adv=(k+(h-k)*rg)/W*N/2
        diag=(h-k)/W-2*diff
        low=diff-adv
        high=diff+adv
        rhs=p.copy()
        if theta!=1:
            alpha=(1-theta)*dt
            rhs+=alpha*diag*p
            rhs[1:]+=alpha*low[1:]*p[:-1]
            rhs[:-1]+=alpha*high[:-1]*p[1:]
        band=np.zeros((3,N-1))
        band[0,1:]=-theta*dt*high[:-1]
        band[1]=1-theta*dt*diag
        band[2,:-1]=-theta*dt*low[1:]
        return solve_banded((1,1),band,rhs)
    def advance(p,left,right):
        steps=max(8,round((right-left)*N*1.7))
        dt=(right-left)/steps
        for j in range(steps):
            at=left+(j+.5)*dt
            if j==0:
                p=step(p,at-dt/4,dt/2,1)
                p=step(p,at+dt/4,dt/2,1)
            else:
                p=step(p,at,dt,.5)
        return p
    def mass(gates):
        p=np.zeros(N-1);p[-1]=1.
        prev=a
        for x,side,lo,hi in sorted(gates):
            p=advance(p,prev,x)
            speed=rg*width(x) if side=="forward" else (1-rg)*width(x)
            cell=width(x)/N
            coverage=np.maximum(0,np.minimum(speed+cell/2,hi)-np.maximum(speed-cell/2,lo))/cell
            p*=coverage
            prev=x
        return advance(p,prev,b)[0]
    return mass(q["gates"])/mass(q["gates"][:1])
