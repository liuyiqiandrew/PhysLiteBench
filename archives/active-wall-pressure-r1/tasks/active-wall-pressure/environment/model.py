import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from functools import lru_cache


@lru_cache(maxsize=256)
def _moments(v,T,L,h,nx):
    Dr,k,ell,modes,tail=.8,4.,.6,18,3.
    x=np.linspace(-L-tail,L+tail,nx);dx=x[1]-x[0]
    n=np.arange(-modes,modes+1);nm=len(n);ids=np.arange(nx*nm).reshape(nx,nm)
    rows=[];cols=[];values=[]
    def add(r,c,z):
        rows.extend(np.ravel(r).tolist());cols.extend(np.ravel(c).tolist());values.extend(np.ravel(z).tolist())
    xf=.5*(x[1:]+x[:-1]);sf=np.maximum(abs(xf)-L,0)
    ap=4*sf**3*ell**4/(ell**4+sf**4)**2
    coeff={0:-np.sign(xf)*(k*sf**3+h*ap/2),1:np.full(nx-1,v/2),-1:np.full(nx-1,v/2),2:np.sign(xf)*h*ap/4,-2:np.sign(xf)*h*ap/4}
    for j,nout in enumerate(n):
        for shift,a in coeff.items():
            q=j-shift
            if not 0<=q<nm:continue
            for source_side in [0,1]:
                flux=a/2 + (T/dx if source_side==0 and shift==0 else 0) - (T/dx if source_side==1 and shift==0 else 0)
                add(ids[:-1,j],ids[source_side:nx-1+source_side,q],-flux/dx)
                add(ids[1:,j],ids[source_side:nx-1+source_side,q],flux/dx)
    s=np.maximum(abs(x)-L,0);a=s**4/(ell**4+s**4)
    for j,nout in enumerate(n):
        add(ids[:,j],ids[:,j],np.full(nx,-Dr*nout*nout))
        for shift,sign in [(2,1),(-2,-1)]:
            q=j-shift
            if 0<=q<nm:add(ids[:,j],ids[:,q],sign*nout*Dr*h*a/(2*T))
    A=coo_matrix((values,(rows,cols)),shape=(nx*nm,)*2).tocsc()
    center=nx//2;anchor=ids[center,modes]
    Q=A.tolil();Q[anchor,:]=0;Q[anchor,anchor]=1;rhs=np.zeros(nx*nm);rhs[anchor]=1
    c=spsolve(Q.tocsc(),rhs).reshape(nx,nm);c/=c[:,modes].sum()*dx
    return x,c[:,modes:]

def _pressure_on_grid(speed,temperature,half_width,alignment,nx):
    x,c=_moments(speed,temperature,half_width,alignment,nx)
    dx=x[1]-x[0]
    mid=len(x)//2
    gradient=(c[mid+1,1]-c[mid-1,1])/(2*dx)
    value=temperature*c[mid,0]+speed/.8*(speed*(c[mid,0]+c[mid,2])/2-temperature*gradient)
    return float(value)


@lru_cache(maxsize=256)
def unit_pressure(speed,temperature,half_width,alignment):
    coarse=_pressure_on_grid(speed,temperature,half_width,alignment,129)
    fine=_pressure_on_grid(speed,temperature,half_width,alignment,257)
    return (4*fine-coarse)/3


def predict_at(experiments,loading):
    return np.asarray([float(loading)*unit_pressure(*[float(e[key]) for key in
                       ('speed','temperature','half_width','alignment')])
                       for e in experiments],dtype=float)


class Model:
    def __init__(self):
        self.loading=None

    def fit(self,records):
        raise NotImplementedError('Fit the common line loading from the calibration records.')

    def predict(self,experiments):
        return predict_at(experiments,self.loading)
