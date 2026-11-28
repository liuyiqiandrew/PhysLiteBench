"""Multiply the spin operators in a sparse Clifford basis, then take their thermal trace."""
from functools import lru_cache

import numpy as np
from scipy.fft import dct


TRUE_PARAMETER = .8364


def measurement(q):
    n = q["n"]
    active = sorted({j for arm in q["arms"] for a,b,_,_ in arm for j in (a,b)})
    basis = np.eye(2*n) if q["frame"] == "site" else dct(np.eye(2*n), axis=0, norm="ortho")
    basis = basis[active]
    temp = np.array([np.tanh(q["beta"]*(1+q["ripple"]*np.cos((2*j+1)*np.pi/n))/2)
                     for j in range(n)])
    parity_weight = q["parity"]*np.prod(temp)

    def moments(t):
        half = (basis[:, ::2] * t) @ basis[:, 1::2].T
        two = 1j*(half-half.T)

        @lru_cache(None)
        def wick(mask):
            if mask == 0:
                return 1.
            first = mask & -mask
            rest = mask ^ first
            options = rest
            sign = 1
            total = 0j
            while options:
                bit = options & -options
                total += sign * two[first.bit_length()-1,bit.bit_length()-1] * wick(rest ^ bit)
                options ^= bit
                sign = -sign
            return total
        return wick

    thermal, twisted = moments(temp), moments(1/temp)

    @lru_cache(None)
    def charge_average(mask):
        return (thermal(mask) + parity_weight*twisted(mask)) / (1+parity_weight)

    @lru_cache(None)
    def left_pair(mask, a, b):
        swaps = (mask & ((1 << b)-1)).bit_count()
        reduced = mask ^ (1 << b)
        swaps += (reduced & ((1 << a)-1)).bit_count()
        return reduced ^ (1 << a), -1 if swaps % 2 else 1

    def trace_product(pulses):
        state = {0: 1.}
        for a,b,theta in pulses:
            c, s = np.cos(.5*theta), -np.sin(.5*theta)
            new = {m: c*v for m,v in state.items()}
            for m,v in state.items():
                out, sign = left_pair(m, a, b)
                new[out] = new.get(out,0) + sign*s*v
            state = new
        return sum(value*charge_average(mask) for mask,value in state.items())

    arms = [[[(active.index(a), active.index(b), th+z*k) for a,b,th,k in arm]
             for z in (1,-1)] for arm in q["arms"]]
    weight = [1/2, q["outcome"]*np.exp(-1j*q["meter_phase"])/2]

    def overlap(first, second):
        total = 0j
        for i, path in enumerate(first):
            backward = [(a,b,-th) for a,b,th in reversed(path)]
            for j, forward in enumerate(second):
                total += weight[i].conjugate()*weight[j]*trace_product(forward+backward)
        return total

    pa, pb = (overlap(arm, arm).real for arm in arms)
    acceptance = (pa+pb)/2
    cross = overlap(*arms)
    signal = (np.exp(-1j*q["phase"])*cross).real/acceptance
    return acceptance, signal


def dense(q):
    """Evolve explicit occupation states of both qubits and the island."""
    n = q["n"]
    I = np.eye(2)
    X = np.array([[0.,1.],[1.,0.]])
    Y = np.array([[0.,-1j],[1j,0.]])
    Z = np.diag([1.,-1.])
    g = []
    for j in range(n):
        for middle in (X,Y):
            op = np.ones((1,1))
            for f in [Z]*j+[middle]+[I]*(n-j-1):
                op = np.kron(op, f)
            g.append(op)
    g = np.array(g)
    diagonal = np.array([np.diag(-1j*g[2*j]@g[2*j+1]).real for j in range(n)])
    energies = 1 + q["ripple"]*np.cos(np.pi*(2*np.arange(n)+1)/n)
    w = np.exp(q["beta"]*(energies @ diagonal)/2)
    w[np.prod(diagonal,axis=0) != q["parity"]] = 0.
    w /= sum(w)
    if q["frame"] == "wave":
        g = dct(g,axis=0,norm="ortho")
    states = []
    for arm in q["arms"]:
        psi = np.stack([np.diag(np.sqrt(w/2)).astype(complex) for _ in range(2)])
        for a,b,th,coupling in arm:
            pair = -1j*g[a]@g[b]
            for meter in (0,1):
                angle=(th+(-1)**meter*coupling)/2
                psi[meter]=np.cos(angle)*psi[meter] - 1j*np.sin(angle)*pair@psi[meter]
        states.append((psi[0]+q["outcome"]*np.exp(-1j*q["meter_phase"])*psi[1])/np.sqrt(2))
    prob = sum(np.vdot(x,x).real for x in states)/2
    return prob, (np.exp(-1j*q["phase"])*np.vdot(*states)).real/prob
