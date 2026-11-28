import math
from functools import lru_cache

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


@lru_cache(maxsize=32)
def oriented_sheet(w, h, holes, vertical, diagonal):
    absent = set(holes)
    sites = [(x,y) for y in range(h) for x in range(w) if (x,y) not in absent]
    ix = {p:i for i,p in enumerate(sites)}
    # Neighbor order is counterclockwise in the embedding.
    directions = [(1,0),(1,1),(0,1),(-1,0),(-1,-1),(0,-1)]
    neighbors = []
    for x,y in sites:
        neighbors.append([ix[x+dx,y+dy] for dx,dy in directions if (x+dx,y+dy) in ix])
    owner = {}
    faces = []
    area = []
    for i,adj in enumerate(neighbors):
        for j in adj:
            if (i,j) in owner:
                continue
            a,b = i,j
            face = []
            while (a,b) not in owner:
                owner[a,b] = len(faces)
                face.append((a,b))
                dest = neighbors[b]
                a,b = b,dest[dest.index(a)-1]
            faces.append(face)
            area.append(sum(sites[a][0]*sites[b][1]-sites[b][0]*sites[a][1] for a,b in face))
    outer = int(np.argmin(area))
    # Flip one edge on the route to the outer face whenever a bounded face
    # has an even number of clockwise arrows. Child faces stay fixed.
    tree = {outer:None}
    order = [outer]
    for f in order:
        for a,b in faces[f]:
            g = owner[b,a]
            if g not in tree:
                tree[g] = (min(a,b),max(a,b))
                order.append(g)
    reversed_edges = set()
    for f in reversed(order[1:]):
        clockwise = sum((a>b) ^ ((min(a,b),max(a,b)) in reversed_edges) for a,b in faces[f])
        if clockwise % 2 == 0:
            reversed_edges.add(tree[f])
    rows,cols,entries = [],[],[]
    for a,adj in enumerate(neighbors):
        for b in adj:
            if a >= b:
                continue
            x,y = sites[a]
            xx,yy = sites[b]
            value = diagonal if x!=xx and y!=yy else vertical if x==xx else 1.
            if (a,b) in reversed_edges:
                value = -value
            rows.extend((a,b)); cols.extend((b,a)); entries.extend((value,-value))
    k = coo_matrix((entries,(rows,cols)),shape=(len(ix),len(ix))).tocsc()
    return ix, k, splu(k)


def pfaffian(a):
    if len(a)==0:
        return 1.
    value = 0.
    for j in range(1,len(a)):
        if a[0,j]:
            keep = [k for k in range(1,len(a)) if k!=j]
            value += (-1)**(j+1)*a[0,j]*pfaffian(a[np.ix_(keep,keep)])
    return value


def odd(q,z):
    holes = tuple(sorted(tuple(p) for p in q['vacancies']))
    ix,k,lu = oriented_sheet(q['w'],q['h'],holes,q['v'],z*q['d'])
    selected, changes = [],[]
    steps = {'H':(1,0),'V':(0,1),'D':(1,1)}
    for x,y,d in q['probe']:
        dx,dy = steps[d]
        a,b = ix[x,y], ix[x+dx,y+dy]
        selected.extend((a,b))
        changes.append(-2*k[a,b])
    n = len(selected)
    rhs = np.zeros((len(ix),n))
    rhs[selected,np.arange(n)] = 1.
    small = lu.solve(rhs)[selected,:]
    for i,weight in enumerate(changes):
        small[2*i,2*i+1] -= 1/weight
        small[2*i+1,2*i] += 1/weight
    ratio = math.prod(-v for v in changes)*pfaffian(small)
    return (1-ratio)/2


class Model:
    def fit(self,records):
        self.activity = float(np.mean([math.sqrt(r['value']/(1-r['value']))/r['input']['d']
                                       for r in records]))
        return self

    def predict(self,inputs):
        return np.array([odd(q,self.activity) for q in inputs])
