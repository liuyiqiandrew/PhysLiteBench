from functools import lru_cache
from itertools import combinations
import numpy as np


@lru_cache(maxsize=128)
def sectors(n, setting):
    beta, h, jx, jy = setting
    data = []
    for parity in (1,-1):
        coupling = np.eye(n)*h/2
        for j in range(n):
            k = (j+1) % n
            bond = -parity if k == 0 else 1
            coupling[k,j] -= bond*jx
            coupling[j,k] -= bond*jy
        u, energy, vt = np.linalg.svd(coupling)
        g = (u*np.tanh(beta*energy)) @ vt
        weight = np.prod(2*np.cosh(beta*energy))
        data.append((g,weight,parity))
    return data


def moments(n, setting, basis):
    data = sectors(n,tuple(setting))
    values = np.zeros(len(basis))
    for g,weight,parity in data:
        for i,(a,b,ca,cb,sign) in enumerate(basis):
            value = np.linalg.det(g[np.ix_(a,b)])
            complement = np.linalg.det(g[np.ix_(ca,cb)])
            values[i] += weight*(value+parity*sign*complement)
    return values/values[0]


@lru_cache(maxsize=128)
def basis_terms(n, sites):
    terms = []
    all_sites = set(range(n))
    for degree in range(len(sites)+1):
        for aa in combinations(sites,degree):
            for bb in combinations(sites,degree):
                gap = []
                for j in all_sites.difference(sites):
                    if sum(k>j for k in aa+bb) % 2:
                        gap.append(j)
                a, b = sorted((*aa,*gap)), sorted((*bb,*gap))
                ca, cb = sorted(all_sites.difference(a)), sorted(all_sites.difference(b))
                terms.append((a,b,ca,cb,(-1)**(sum(a)+sum(b))))
    return terms


def fringe(q):
    sites = tuple(sorted(q["sites"]))
    basis = basis_terms(q["n"],sites)
    first = moments(q["n"],q["first"],basis)
    second = moments(q["n"],q["second"],basis)
    return (first @ second)/2**len(sites)


class Model:
    def fit(self, records):
        x = np.array([fringe(r["input"])/r["sigma"] for r in records])
        y = np.array([r["value"]/r["sigma"] for r in records])
        self.visibility = float(np.clip((x@y)/(x@x), .65, .95))
        return self

    def predict(self, inputs):
        return self.visibility*np.array([fringe(q) for q in inputs])
