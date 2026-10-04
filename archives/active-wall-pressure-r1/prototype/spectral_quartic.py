"""Independent Fourier orientation moments and centered conservative spatial flux."""
from pathlib import Path
import json
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def solve(Dr=.8,v=1.5,T=.5,L=1.5,k=4,h=0,ell=.6,nx=401,modes=16,tail=3):
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
    ap=4*s**3*ell**4/(ell**4+s**4)**2
    force=np.sum(((k*s**3+h*ap/2)*c[:,modes]-h*ap/2*c[:,modes+2])[center+1:])*dx
    shortcut=T*c[center,modes]+v/Dr*(v*.5*(c[center,modes]+c[center,modes+2])-T*(c[center+1,modes+1]-c[center-1,modes+1])/(2*dx))
    torque=-np.sum((Dr*h*a/(2*T)*(c[:,modes+1]-c[:,modes+3]))[center+1:])*dx
    theta=np.arange(128)*2*np.pi/128
    density=(c@np.exp(1j*n[:,None]*theta[None,:])).real/(2*np.pi)
    return {'force':float(force),'shortcut':float(shortcut),'moment_balance':float(shortcut-v/Dr*torque),'relative_gap':float((shortcut-force)/force),'minimum_density':float(density.min()),'edge_density':float(np.max(abs(density[[0,-1]]))),'normalization':float(c[:,modes].sum()*dx),'stationarity_residual':float(np.max(abs(A@c.ravel()))),'center_density':float(c[center,modes])}


def main():
    report={'scope':'Independent angular Fourier / centered flux discretization; prototype only.','cases':[]}
    for h in [0,1,2,-1]:
        case={'Dr':.8,'v':1.5,'T':.5,'L':1.5,'k':4,'h':h,'ell':.6}
        report['cases'].append({'inputs':case,'coarse':solve(**case,nx=321,modes=12),'fine':solve(**case,nx=641,modes=16)})
    out=Path(__file__).with_name('spectral-quartic-report.json');out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
