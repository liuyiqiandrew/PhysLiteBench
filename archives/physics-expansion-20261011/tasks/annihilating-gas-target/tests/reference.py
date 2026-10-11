"""Complex generating-function inversion and microscopic collision reference."""
import numpy as np
from functools import lru_cache
from scipy.special import gammainc

TRUE_PARAMETER = 1.08


def experiment(density=.5, stationary_fraction=0., time=1.):
    return dict(density=density, stationary_fraction=stationary_fraction, time=time)


def calibration_inputs():
    return [experiment(r, 0., t) for r in [.2, .5, .8, 1.2]
            for t in [.25, .5, .75, 1., 1.5, 2., 2.5, 3.]]


def hidden_inputs():
    return {
        'shared_anchors': [experiment(r, p, t) for r in [.2, 1.2]
                           for p in [0., 1.] for t in [0., 5., 40.]],
        'dilute_stationary': [experiment(r, p, t) for r in [.4, 1.]
                              for p in [.1, .2] for t in [6., 15., 40.]],
        'mixed_gas': [experiment(r, p, t) for r in [.4, 1.]
                      for p in [.3, .45] for t in [6., 15., 40.]],
        'stationary_rich': [experiment(r, p, t) for r in [.4, 1.]
                            for p in [.6, .8] for t in [6., 15., 40.]],
    }


@lru_cache(maxsize=256)
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



def predict(experiments, speed, samples=2048, radius=.98):
    values = []
    for e in experiments:
        z = e['density']*speed*e['time']
        a = fft_coefficients(e['stationary_fraction'], samples=samples, radius=radius)
        q = np.dot(a, gammainc(np.arange(1, len(a)+1), z))
        values.append((1-q)**2)
    return np.asarray(values)


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

