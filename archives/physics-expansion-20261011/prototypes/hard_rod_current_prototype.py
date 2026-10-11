import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import xlogy


def exact(rho,a,p,lam,v=1.):
    c=rho/(1-a*rho)
    s=brentq(lambda s:s+a*c*(p*np.expm1(s)-(1-p)*np.expm1(-s))-lam,-abs(lam)-1,abs(lam)+1,xtol=1e-14)
    return c*v*(p*np.expm1(s)+(1-p)*np.expm1(-s))


def shortcut(rho,a,p,lam,v=1.):
    h=1-a*rho;c=rho/h;d=2*p-1
    rp=c*p*v*(1-a*rho*d);rm=c*(1-p)*v*(1+a*rho*d)
    return rho*v*d*lam+rp*(np.expm1(h*lam)-h*lam)+rm*(np.expm1(-h*lam)+h*lam)


def rate(j,rho,a,p,v=1.):
    c=rho/(1-a*rho)
    rp=c*p*(v-a*j);rm=c*(1-p)*(v+a*j)
    z=np.sqrt(j*j+4*rp*rm)
    plus=(z+j)/2;minus=(z-j)/2
    return xlogy(plus,plus/rp)-plus+rp+xlogy(minus,minus/rm)-minus+rm


def contraction(rho,a,p,lam,v=1.):
    if a==0:return rho*v*(p*np.expm1(lam)+(1-p)*np.expm1(-lam))
    opt=minimize_scalar(lambda j:rate(j,rho,a,p,v)-lam*j,bounds=(-v/a+1e-9,v/a-1e-9),method='bounded',options={'xatol':1e-12})
    return -opt.fun

if __name__=='__main__':
    for rho,a,p in [(.5,1.,.5),(.6,1.2,.5),(.5,1.,.3),(.5,1.,.7),(.3,.8,.5)]:
        for lam in [-4,-2,2,4]:
            f=exact(rho,a,p,lam);g=shortcut(rho,a,p,lam);r=contraction(rho,a,p,lam)
            print(rho,a,p,lam,f,g,abs(g/f-1),abs(f-r))


def discrete_scgf(rho,a,p,lam,t,v=1.):
    from scipy.special import ive,logsumexp,gammaln
    c=rho/(1-a*rho)
    def tails(k):
        z=a*(k-1)
        rp=c*(p*max(v*t-z,0.)+(1-p)*max(-v*t-z,0.))
        rm=c*(p*max(z-v*t,0.)+(1-p)*max(z+v*t,0.))
        if rp==0 and k>0:return -np.inf,0.
        if rm==0 and k<=0:return 0.,-np.inf
        mean=rp-rm;sd=np.sqrt(rp+rm)
        lo=int(np.floor(min(mean,k)-max(80,15*sd)));hi=int(np.ceil(max(mean,k)+max(80,15*sd)))
        n=np.arange(lo,hi+1)
        if rp==0:
            logs=np.where(n<=0,-rm-n*np.log(rm)-gammaln(-n+1),-np.inf)
        elif rm==0:
            logs=np.where(n>=0,-rp+n*np.log(rp)-gammaln(n+1),-np.inf)
        else:
            z=2*np.sqrt(rp*rm)
            with np.errstate(divide='ignore'):
                logs=-(np.sqrt(rp)-np.sqrt(rm))**2+.5*n*np.log(rp/rm)+np.log(ive(abs(n),z))
        return logsumexp(logs[n>=k]),logsumexp(logs[n<k])
    bound=int(np.ceil(v*t/a))+2
    ks=np.arange(-bound,bound+1)
    tails_by_k=[tails(int(k)) for k in ks]
    terms=[]
    for i,k in enumerate(ks[:-1]):
        upper,lower=tails_by_k[i];next_upper,next_lower=tails_by_k[i+1]
        if k>=rho*v*(2*p-1)*t:
            if not np.isfinite(upper):continue
            logprob=upper+np.log(-np.expm1(min(0.,next_upper-upper)))
        else:
            if not np.isfinite(next_lower):continue
            logprob=next_lower+np.log(-np.expm1(min(0.,lower-next_lower)))
        terms.append(lam*k+logprob)
    return logsumexp(terms)/t
