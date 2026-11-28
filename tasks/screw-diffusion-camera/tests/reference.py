"""Approximate the generator by advancing complete finite screws at the material origin."""
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import special
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.spatial.transform import Rotation


TRUE_PARAMETER = .9138
rates = [.55,.82,1.12]
screws = np.array([[.27,.46,-.18],[.04,-.23,.33],[-.38,.17,.29]])


def harmonics(points, order):
    longitude = np.arctan2(points[:,1],points[:,0])
    latitude = np.arccos(np.clip(points[:,2],-1.,1.))
    labels = [(l,m) for l in range(order+1) for m in range(-l,l+1)]
    if hasattr(special, "sph_harm_y"):
        return np.column_stack([special.sph_harm_y(l,m,latitude,longitude) for l,m in labels])
    return np.column_stack([special.sph_harm(m,l,longitude,latitude) for l,m in labels])


@lru_cache(maxsize=4)
def grid(order, step):
    nz = order+7
    nphi = 2*nz+1
    z,w = leggauss(nz)
    longitude = np.tile(2*np.pi*np.arange(nphi)/nphi, nz)
    z = np.repeat(z,nphi)
    xy = np.sqrt(1-z*z)
    u = np.column_stack([xy*np.cos(longitude),xy*np.sin(longitude),z])
    weight = np.repeat(w*2*np.pi/nphi,nphi)
    y = harmonics(u,order)
    basis = []
    for axis,screw in zip(np.eye(3),screws):
        parallel = (axis@screw)*axis
        pair = []
        for h in (step,-step):
            end = u @ Rotation.from_rotvec(h*axis).as_matrix()
            displacement = h*parallel+np.sin(h)*(screw-parallel)+(1-np.cos(h))*np.cross(axis,screw)
            pair.append((harmonics(end,order),u@displacement))
        basis.append(pair)
    return u,weight,y,y.conj().T*weight,basis


@lru_cache(maxsize=128)
def generator(order, step, k, hand):
    u,_,y,adj,neighbors = grid(order,step)
    diff = -(k*k)*(u**2 @ [.18,.27,.36])[:,None]*y
    for r, shifted in zip(rates,neighbors):
        forward, backward = [np.exp(1j*k*hand*dist)[:,None]*basis for basis,dist in shifted]
        diff += r*(forward+backward-2*y)/(step**2)
    matrix = adj @ diff
    matrix[abs(matrix)<3e-9] = 0.
    return csr_matrix(matrix)


def measurement(q, mobility=TRUE_PARAMETER, order=9, step=.003):
    u,w,y,adj,_=grid(order,step)
    tag = u@q["tag"]
    gates = [(1+u@g)**2/4 for g in q["gates"]]
    def advance(coeff,k,t):
        matrix = generator(order,step,k,q["hand"])
        return expm_multiply((mobility*t)*matrix,coeff)
    k1,k2=q["wave"]
    t1,t2=q["times"]
    # All propagation follows the material origin. Tag offsets enter at the three exposures.
    v = advance(adj@np.exp(1j*k2*tag),k2,t2)
    middle=adj@(gates[1]*np.exp(1j*(k1-k2)*tag)*(y@v))
    v = advance(middle,k1,t1)
    num = (w*gates[0]*np.exp(-1j*k1*tag))@(y@v)
    norm = (w*gates[0])@(y@advance(adj@gates[1],0,t1))
    return float((np.exp(-1j*q["phase"])*num/norm).real)
