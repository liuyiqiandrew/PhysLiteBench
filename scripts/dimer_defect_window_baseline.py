from functools import lru_cache
import math
import numpy as np

STEPS = {'H': (1, 0), 'V': (0, 1), 'D': (1, 1)}


def patch_occupation(q, bond, activity):
    x, y, direction = bond
    dx, dy = STEPS[direction]
    end = (x+dx, y+dy)
    excluded = {tuple(site) for site in q['vacancies']}
    weights = {'H': 1., 'V': q['v'], 'D': activity*q['d']}

    def present(a, b):
        return 0 <= a < q['w'] and 0 <= b < q['h'] and (a, b) not in excluded

    sites = [(a,b) for a in range(x-1, end[0]+2) for b in range(y-1, end[1]+2)
             if present(a,b)]
    index = {site:i for i,site in enumerate(sites)}
    neighbors = [[] for site in sites]
    reservoir = [0.] * len(sites)
    for i,(a,b) in enumerate(sites):
        for name,(u,v) in STEPS.items():
            for sign in (-1,1):
                other = (a+sign*u, b+sign*v)
                if other in index:
                    neighbors[i].append((index[other],weights[name]))
                elif present(*other):
                    reservoir[i] += weights[name]

    @lru_cache(None)
    def total(mask):
        if mask == 0:
            return 1.
        bit = mask & -mask
        i = bit.bit_length()-1
        rest = mask ^ bit
        value = reservoir[i] * total(rest) if reservoir[i] else 0.
        for j,weight in neighbors[i]:
            if rest & (1 << j):
                value += weight * total(rest ^ (1 << j))
        return value

    all_sites = (1 << len(sites))-1
    free = all_sites ^ (1 << index[(x,y)]) ^ (1 << index[end])
    return weights[direction] * total(free) / total(all_sites)


class Model:
    def fit(self, records):
        self.activity = float(np.mean([math.sqrt(r['value']/(1-r['value']))/r['input']['d']
                                       for r in records]))
        return self

    def predict(self, inputs):
        return np.array([(1 - math.prod(1-2*patch_occupation(q,b,self.activity)
                                       for b in q['probe']))/2 for q in inputs])
