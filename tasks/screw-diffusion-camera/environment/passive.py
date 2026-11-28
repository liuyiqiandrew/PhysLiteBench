"""Backward scattering propagator at the tracked material point."""
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import gammaln, lpmv
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import expm_multiply

from apparatus import ROT, SCREWS, TRANS


@lru_cache(maxsize=4)
def operators(order):
    # Keep one extra angular degree for the two successive coordinate multipliers.
    L = order + 1
    n = (L+1)**2
    z, w = leggauss(L+2)
    nphi = 2*L+3
    phi = np.tile(np.arange(nphi)*2*np.pi/nphi,len(z))
    z = np.repeat(z,nphi)
    weight = np.repeat(w*2*np.pi/nphi,nphi)
    unit = np.array([np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z])
    Y = np.empty((len(z),n), complex)
    raise_m = np.zeros((n,n))
    magnetic = np.zeros(n)
    for l in range(L+1):
        for m in range(-l,l+1):
            i = l*l+l+m
            a = abs(m)
            norm = np.sqrt((2*l+1)/(4*np.pi)*np.exp(gammaln(l-a+1)-gammaln(l+a+1)))
            y = norm*lpmv(a,l,z)*np.exp(1j*a*phi)
            Y[:,i] = (-1)**a*y.conj() if m<0 else y
            magnetic[i] = m
            if m<l:
                raise_m[i+1,i] = np.sqrt((l-m)*(l+m+1))
    pos = []
    for v in unit:
        mat = Y.conj().T @ ((weight*v)[:,None]*Y)
        mat[abs(mat)<2e-13]=0.
        pos.append(csr_matrix(mat))
    spin = [(-.5j)*(raise_m+raise_m.T), -.5*(raise_m-raise_m.T), -1j*np.diag(magnetic)]
    return pos, [csr_matrix(g) for g in spin]


def project(matrices, v):
    return sum(a*m for a,m in zip(v,matrices))


def fringe(q, mobility, order=7):
    pos, spin = operators(order)
    count = (order+1)**2
    one = np.zeros(count,complex)
    one[0] = 1
    gates = []
    for axis in q["gates"]:
        factor = eye(pos[0].shape[0],format="csr")+project(pos,axis)
        gates.append((factor @ factor)[:count,:count]/4)
    tag = np.asarray(q["tag"])
    screw = q["hand"]*SCREWS

    def evolve(v,k,t):
        G = -k*k*sum(d*(p@p) for d,p in zip(TRANS,pos))
        for j,r in enumerate(ROT):
            lever = np.cross(np.eye(3)[j],tag)
            lever = lever+screw[j]
            twist = spin[j] + (1j*k)*project(pos,lever)
            G += r*(twist@twist)
        return expm_multiply((mobility*t)*G[:count,:count],v)

    times = q["times"]
    first, second = q["wave"]
    right = evolve(one,second,times[1])
    signal = (gates[0] @ evolve(gates[1]@right,first,times[0]))[0]
    accepted = (gates[0] @ evolve(gates[1]@one,0,times[0]))[0].real
    return float((np.exp(-1j*q["phase"])*signal).real/accepted)
