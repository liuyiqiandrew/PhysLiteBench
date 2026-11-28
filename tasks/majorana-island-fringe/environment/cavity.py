"""Conditioned coherent wave packet from a parity-driven cavity."""
from itertools import product

import numpy as np
from scipy.special import ndtr

from apparatus import contacts, polarizations
from island_trace import pfaffian


def spin_function(q, parity):
    frame = contacts(q)
    t = polarizations(q)
    charge = parity*np.prod(t) if parity else 0.
    def cov(temp):
        half = (frame[:,::2]*temp) @ frame[:,1::2].T
        return np.eye(len(frame)) + 1j*(half-half.T)
    states = [(cov(t),1.)]
    if parity:
        states.append((cov(1/t),charge))

    def evaluate(left, right):
        # Lists are chronological. The operator here is left† right.
        operators = [(a,b,c.conjugate(),-s.conjugate()) for a,b,c,s in left]+right[::-1]
        labels = [v for a,b,_,_ in operators for v in (a,b)]
        even = np.arange(0,len(labels),2)
        c = [f[2] for f in operators]
        scale = np.ones(len(labels),complex)
        scale[even] = [f[3] for f in operators]
        value = 0j
        for covariance, weight in states:
            upper = np.triu(covariance[np.ix_(labels,labels)]*scale[:,None]*scale[None,:],1)
            upper[even,even+1] += c
            value += weight*pfaffian(upper-upper.T)
        return value/(1+charge)
    return evaluate


def paths(q, arm):
    cavity = q["cavity"]
    bins = cavity["segments"][arm]
    output = []
    for meter in (1,-1):
        for signs in product((1,-1), repeat=len(bins)):
            d = np.zeros(len(bins)+1,complex)
            phase = 0.
            factors = []
            start = 0
            for j, ((end,a,b,x,y), sig, eta) in enumerate(zip(bins,signs,cavity["transmission"])):
                for i,k,theta,coupling in q["arms"][arm][start:end]:
                    half = (theta+meter*coupling)/2
                    factors.append((i,k,np.cos(half),-np.sin(half)))
                start=end
                impulse = sig*complex(x,y)
                factors.append((a,b,.5,-.5j*sig))
                phase += (d[0].conjugate()*impulse).imag
                d[0] += impulse
                d[j+1] = -np.sqrt(1-eta)*d[0]
                d[0] *= np.sqrt(eta)
            m = 1 if meter==1 else q["outcome"]*np.exp(-1j*q["meter_phase"])
            output.append((m*np.exp(1j*phase)/2,d,factors))
    return output


def cavity_fringe(q, parity, retain_window=True):
    c = q["cavity"]
    v = np.zeros(len(c["transmission"])+1)
    v[0] = 1.
    for j, eta in enumerate(c["transmission"]):
        v[j+1] = -np.sqrt(1-eta)*v[0]
        v[0] *= np.sqrt(eta)
    g = np.r_[0j, [complex(*w) for w in c["mode"]]]
    g *= np.exp(1j*c["angle"])/np.linalg.norm(g)
    M = np.eye(len(v)) + 2*c["thermal"]*np.outer(v,v)
    sd = np.sqrt(np.vdot(g,M@g).real/2)
    lo,hi = c["window"]
    spin = spin_function(q,parity)

    def overlap(left,right):
        total = 0j
        for wa, a, fa in left:
            for wb, b, fb in right:
                delta = b-a
                shift = np.vdot(g,a+b).real/np.sqrt(2) + 1j*np.vdot(g,M@delta).imag/np.sqrt(2)
                all_photons = np.exp(1j*np.vdot(a,b).imag - np.vdot(delta,M@delta).real/2)
                bin_weight = ndtr((hi-shift)/sd)-ndtr((lo-shift)/sd) if retain_window else 1.
                total += wa.conjugate()*wb*all_photons*bin_weight*spin(fa,fb)
        return total
    first, second = paths(q,0), paths(q,1)
    accept=(overlap(first,first)+overlap(second,second)).real/2
    cross=overlap(first,second)
    return accept, float((np.exp(-1j*q["phase"])*cross).real/accept)
