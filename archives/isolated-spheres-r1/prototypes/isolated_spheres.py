import json
import numpy as np
from scipy.special import gammaln


def multipoles(distance, radius=1., isolated=False, cutoff=32):
    energy=force=0.
    mineig=1.
    for m in range(cutoff+1):
        ell=np.arange(max(m,int(isolated)),cutoff+1)
        if not len(ell):continue
        l=ell[:,None];q=ell[None,:]
        power=l+q+1
        logs=gammaln(l+q+1)-.5*(gammaln(l+m+1)+gammaln(l-m+1)+gammaln(q+m+1)+gammaln(q-m+1))
        matrix=np.exp(logs+power*np.log(radius/distance))
        derivative=-power*matrix/distance
        det=np.eye(len(ell))-matrix@matrix
        energy+=.5*(1 if m==0 else 2)*np.linalg.slogdet(det)[1]
        force+=(1 if m==0 else 2)*np.trace(np.linalg.solve(det,matrix@derivative))
        mineig=min(mineig,float(np.linalg.eigvalsh(det)[0]))
    return energy,force,mineig


def bispherical(distance,radius=1.,isolated=False,cutoff=300):
    p=np.arccosh(distance/(2*radius))
    dp=1/np.sqrt(distance*distance-4*radius*radius)
    n=np.arange(1,cutoff+1,dtype=float)
    odd=2*n-1
    z=np.exp(-2*odd*p)
    energy=.5*np.sum(odd*np.log1p(-z))
    force=-np.sum(odd*odd*z/(1-z))*dp
    if isolated:
        def sc(k):
            arg=k*p
            inv=2*np.exp(-arg)/(1-np.exp(-2*arg))
            s=np.sum(inv)
            ds=-np.sum(k*inv/np.tanh(arg))
            return radius*np.sinh(p)*s,radius*(np.cosh(p)*s+np.sinh(p)*ds)*dp
        a,da=sc(odd);b,db=sc(2*n);b=-b;db=-db
        det=a*a-b*b
        energy+=.5*np.log(det/(radius*radius))
        force-= (a*da-b*db)/det
    return energy,force

if __name__=='__main__':
    data=[]
    for distance in [2.5,2.6,2.8,3.,3.5,4.,5.,8.]:
        g=multipoles(distance,cutoff=40)
        i=multipoles(distance,isolated=True,cutoff=40)
        bg=bispherical(distance);bi=bispherical(distance,isolated=True)
        data.append(dict(distance=distance,grounded=g,isolated=i,bispherical_grounded=bg,bispherical_isolated=bi,force_relative_gap=(g[1]-i[1])/i[1]))
    print(json.dumps(data,indent=2))
