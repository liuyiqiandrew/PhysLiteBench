"""Ordered-radial finite partition integrals and full angular Gibbs checks."""
from functools import lru_cache
import numpy as np
from scipy.linalg import expm
from scipy.special import logsumexp
from scipy.stats import qmc

TRUE_PARAMETER = 1.12


def experiment(multiplier=.2,count=3,radius=1.):
    return dict(temperature_multiplier=multiplier,ion_count=count,disk_radius=radius)


def calibration_inputs():
    return [experiment(u,n,a) for u in [.1,.2,.3] for n in [1,2,3,4] for a in [.7,1.6]]


def hidden_inputs():
    return {
        'shared_anchors': [experiment(u,1,a) for u in [.4,.9,1.6,1.8] for a in [.5,2.]],
        'few_free_ions': [experiment(u,n,a) for u in [.85,.9,.95] for n in [3,4] for a in [.5,2.]],
        'strong_binding': [experiment(u,n,a) for u in [1.25,1.35,1.45] for n in [2,3,4] for a in [.5,2.]],
        'last_free_ion': [experiment(u,n,a) for u in [1.5,1.6,1.7] for n in [2,3,4] for a in [.5,2.]],
    }


def radial(gamma,count,length,return_log_partition=False):
    # Ordered t_i=log(r_i/a), angular correction has no leading log exponent.
    slopes=2-gamma*np.arange(count,0,-1)
    sums=np.r_[0,np.cumsum(slopes)]
    generator=np.diag(-sums)+np.diag(np.ones(count),-1)
    shift=max(-sums)
    generator-=shift*np.eye(count+1)
    initial=np.zeros(count+1);initial[0]=1.
    state=expm(length*generator)@initial
    if return_log_partition:
        return (sums[-1]+shift)*length+np.log(state[-1])
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



@lru_cache(maxsize=None)
def limiting(gamma,count,length=3200.):
    scales = np.array([1.,.5,.25,.125])
    values = [radial(gamma,count,length/s) for s in scales]
    matrix = np.column_stack([np.ones(4),scales,scales**2,scales**3])
    return np.linalg.solve(matrix,values)[0]


def predict(experiments,coupling,length=3200.):
    return np.asarray([limiting(e['temperature_multiplier']*coupling,e['ion_count'],length)
                       for e in experiments])
