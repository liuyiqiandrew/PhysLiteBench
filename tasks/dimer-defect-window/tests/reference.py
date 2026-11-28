"""Positive matching minors with vacancy disorder lines on the full lattice."""
from itertools import combinations
from functools import lru_cache
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

TRUE_PARAMETER = 0.937
DIRECTIONS = {'H': (1,0), 'V': (0,1), 'D': (1,1)}


@lru_cache(maxsize=800)
def log_partition(w,h,holes,v,diag):
    # Orient a regular sheet. Moving an interior vacancy to the outside
    # through the dual graph flips the arrows crossed by a ray to +x.
    cut = set(holes)
    vertices = {(x,y):i for i,(x,y) in enumerate((x,y) for y in range(h)
                for x in range(w) if (x,y) not in cut)}
    if not vertices:
        return 0.
    row,col,values = [],[],[]
    for (x,y),i in vertices.items():
        for d,(dx,dy) in DIRECTIONS.items():
            end = x+dx,y+dy
            if end not in vertices:
                continue
            j = vertices[end]
            weight = {'H':1.,'V':v*(-1)**x,'D':diag*(-1)**(x+1)}[d]
            if dy:
                # Each ray starts at (hx+.2,hy+.25).
                crossings = sum(hy==y and hx+.2 < x+.25*dx for hx,hy in holes)
                weight *= (-1)**crossings
            row.extend((i,j));col.extend((j,i));values.extend((weight,-weight))
    a = coo_matrix((values,(row,col)),shape=(len(vertices),len(vertices))).tocsc()
    try:
        return .5 * np.log(np.abs(splu(a).U.diagonal())).sum()
    except RuntimeError:
        return -math.inf


def probability(q,z=TRUE_PARAMETER):
    holes = frozenset(tuple(p) for p in q['vacancies'])
    v,d = q['v'],z*q['d']
    def free_energy(absent):
        return log_partition(q['w'],q['h'],tuple(sorted(absent)),v,d)
    base = free_energy(holes)
    odd = 0.
    for size in range(1,len(q['probe'])+1):
        for subset in combinations(q['probe'],size):
            endpoints = set()
            product = 1.
            for x,y,label in subset:
                dx,dy = DIRECTIONS[label]
                endpoints.update(((x,y),(x+dx,y+dy)))
                product *= {'H':1.,'V':v,'D':d}[label]
            if len(endpoints)==2*size:
                joint = product*math.exp(free_energy(holes|endpoints)-base)
                odd += (-2)**(size-1)*joint
    return float(odd)


def enumerate_parity(q,z=TRUE_PARAMETER):
    absent = {tuple(p) for p in q['vacancies']}
    sites = sorted((x,y) for x in range(q['w']) for y in range(q['h']) if (x,y) not in absent)
    marks = {tuple(p) for p in q['probe']}
    neighbor = [[] for p in sites]
    for a,(x,y) in enumerate(sites):
        for d,(dx,dy) in DIRECTIONS.items():
            end = x+dx,y+dy
            if end in sites:
                b = sites.index(end)
                weight = {'H':1.,'V':q['v'],'D':z*q['d']}[d]
                neighbor[a].append((b,weight,(x,y,d) in marks))
    @lru_cache(None)
    def count(mask):
        if not mask:
            return 1.,0.
        bit = mask & -mask
        even = odd = 0.
        for j,weight,marked in neighbor[bit.bit_length()-1]:
            if mask & (1<<j):
                e,o = count(mask^bit^(1<<j))
                if marked:
                    e,o = o,e
                even += weight*e; odd += weight*o
        return even,odd
    a,b = count((1<<len(sites))-1)
    return b/(a+b)
