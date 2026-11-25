import numpy as np
from scipy.linalg import expm
from scipy.special import logsumexp
from scipy.stats import qmc


def exact(gamma,count):
    k=np.arange(count+1)
    return np.max(k-gamma*k*(k+1)/4)/count


def shortcut(gamma,count):
    if count==1:
        return max(1-gamma/2,0.)
    number=np.clip((2-gamma)/(gamma*(1-1/count)),0,count)
    return ((1-gamma/2)*number-gamma*(1-1/count)*number**2/4)/count


def radial(gamma,count,length):
    # Ordered t_i=log(r_i/a), angular correction has no leading log exponent.
    slopes=2-gamma*np.arange(count,0,-1)
    sums=np.r_[0,np.cumsum(slopes)]
    generator=np.diag(-sums)+np.diag(np.ones(count),-1)
    shift=max(-sums)
    generator-=shift*np.eye(count+1)
    initial=np.zeros(count+1);initial[0]=1.
    state=expm(length*generator)@initial
    return (sums[-1]+shift+(generator@state)[-1]/state[-1])/(2*count)


def log_partition(gamma,count,length,power,seed):
    points=qmc.Sobol(2*count,scramble=True,seed=seed).random_base2(power)
    coordinate=points[:,:count]*3
    component=np.floor(coordinate).astype(int)
    u=coordinate-component
    inner=-np.log1p(-u*(-np.expm1(-length)))
    t=np.where(component==0,u*length,np.where(component==1,inner,length-inner))
    theta=points[:,count:]*2*np.pi
    density=(1/length+(np.exp(-t)+np.exp(t-length))/(-np.expm1(-length)))/3
    weight=(2-gamma*count)*np.sum(t,axis=1)-np.sum(np.log(density),axis=1)
    for i in range(count):
        for j in range(i):
            decay=np.exp(-abs(t[:,i]-t[:,j]))
            angular=np.log1p(decay*decay-2*decay*np.cos(theta[:,i]-theta[:,j]))/2
            weight+=gamma*(np.maximum(t[:,i],t[:,j])+angular)
    return logsumexp(weight)-np.log(len(weight))


if __name__=='__main__':
    for count,gamma in [(4,.3),(2,1.2),(3,1.6),(4,1.8),(4,2.2),(4,1.)]:
        slopes=[]
        for seed in range(4):
            values=[log_partition(gamma,count,L,18,719001+seed) for L in [40.,80.]]
            slopes.append((values[1]-values[0])/(80*count))
        print(count,gamma,'exact',exact(gamma,count),'shortcut',shortcut(gamma,count),
              'radial',radial(gamma,count,400.),'fullQMC',np.mean(slopes),'std',np.std(slopes,ddof=1),flush=True)
