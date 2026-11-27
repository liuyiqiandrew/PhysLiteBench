"""Spin Hilbert space for generic small rings and Ising transfer matrices for large rings."""
from functools import lru_cache
from itertools import product
import numpy as np

TRUE_PARAMETER = .817


@lru_cache(maxsize=64)
def thermal(n, setting):
    beta,h,jx,jy = setting
    states = np.arange(2**n)
    spin = np.array([1-2*((states>>j)&1) for j in range(n)])
    ham = np.diag(-.5*h*spin.sum(axis=0))
    for j in range(n):
        k = (j+1)%n
        target = states ^ (1<<j) ^ (1<<k)
        ham[target,states] += -jx+jy*spin[j]*spin[k]
    e,v = np.linalg.eigh(ham)
    weight = np.exp(-beta*(e-e.min()))
    return (v*weight) @ v.T / weight.sum()


def reduce_state(n, setting, sites):
    env = sorted(set(range(n)).difference(sites))
    perm = [n-1-j for j in sites+env]
    rho = thermal(n,tuple(setting)).reshape([2]*(2*n)).transpose(perm+[n+j for j in perm])
    m = 2**len(sites)
    return np.trace(rho.reshape(m,2**n//m,m,2**n//m),axis1=1,axis2=3)


def ising_marginal(n, setting, sites):
    beta,h,jx,jy = setting
    assert h == 0 and jx*jy == 0
    axis = "y" if jy else "x"
    coupling = beta*(jx+jy)
    transfer = np.exp(coupling*np.array([[1.,-1.],[-1.,1.]]))
    gaps = np.diff(sites + [n+sites[0]])
    links = [np.linalg.matrix_power(transfer,int(d)) for d in gaps]
    p = []
    for word in product(range(2),repeat=len(sites)):
        p.append(np.prod([t[word[i], word[(i+1)%len(word)]] for i,t in enumerate(links)]))
    return axis, np.array(p)/sum(p)


def fringe(q):
    n, sites = q["n"], sorted(q["sites"])
    if n <= 9:
        a,b = [reduce_state(n,q[key],sites) for key in ["first","second"]]
        return np.sum(a*b.T)
    x,p = ising_marginal(n,q["first"],sites)
    y,s = ising_marginal(n,q["second"],sites)
    return p@s if x == y else 2.**(-len(sites))
