import numpy as np
from scipy.special import gammainc, ive


def coefficients(p, count=256):
    g=np.zeros((count+1)//2);h=np.zeros_like(g)
    g[0]=(1-p)/2;h[0]=g[0]**2
    for k in range(1,len(g)):
        fourth=0. if k==1 else np.dot(h[:k-1],h[k-2::-1])
        g[k]=(1+2*p)/2*h[k-1]-p/2*fourth
        h[k]=np.dot(g[:k+1],g[k::-1])
    result=np.zeros(count);result[::2]=g
    return result


def fft_coefficients(p, count=256, samples=2048, radius=.98):
    contour=radius*np.exp(2j*np.pi*np.arange(samples)/samples)
    f=np.zeros(samples,dtype=complex)
    for fraction in np.linspace(0.,1.,65)[1:]:
        x=fraction*contour
        for _ in range(12):
            residual=p*x*f**4-(1+2*p)*x*f*f+2*f-(1-p)*x
            derivative=4*p*x*f**3-2*(1+2*p)*x*f+2
            f-=residual/derivative
    values=np.fft.fft(f).real/samples
    return values[1:count+1]/radius**np.arange(1,count+1)


def survival(p,z):
    if p==0.:return (ive(0,z)+ive(1,z))**2
    a=coefficients(p)
    return (1-np.dot(a,gammainc(np.arange(1,len(a)+1),z)))**2


if __name__=='__main__':
    for p in [0.,.1,.2,.25,.3,.45,.6,.8,1.]:
        a=coefficients(p);b=fft_coefficients(p)
        print('coefs',p,np.max(abs(a-b)),a.min(),a.sum())
        for z in [5.,15.,40.,90.]:
            y=survival(p,z);shortcut=(ive(0,(1-p)*z)+ive(1,(1-p)*z))**2
            print(p,z,y,shortcut,abs(shortcut/y-1))


def first_hit(positions, velocities, horizon):
    import heapq
    x=np.r_[0.,positions];v=np.r_[0.,velocities]
    n=len(x);alive=np.ones(n,dtype=bool);left=np.arange(n)-1;right=np.arange(n)+1;right[-1]=-1
    events=[]
    def add(i,j):
        if i>=0 and j>=0 and v[i]>v[j]:
            t=(x[j]-x[i])/(v[i]-v[j])
            if t<=horizon:heapq.heappush(events,(t,i,j))
    for i in range(n-1):add(i,i+1)
    while events:
        _,i,j=heapq.heappop(events)
        if not alive[i] or not alive[j] or right[i]!=j:continue
        if i==0:return j
        alive[i]=alive[j]=False
        a,b=left[i],right[j]
        if a>=0:right[a]=b
        if b>=0:left[b]=a
        add(a,b)
    return None


def microscopic(p,z,trials=10000,seed=120):
    rng=np.random.default_rng(seed);hits=0
    for _ in range(trials):
        n=rng.poisson(z);x=np.sort(rng.uniform(0,z,n));v=rng.choice([-1,0,1],n,p=[(1-p)/2,p,(1-p)/2])
        hits+=first_hit(x,v,z) is not None
    q=1-np.sqrt(survival(p,z));observed=hits/trials
    return observed,q,np.sqrt(q*(1-q)/trials)
