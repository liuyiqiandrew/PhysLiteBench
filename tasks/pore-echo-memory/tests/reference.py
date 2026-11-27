"""Finite-time Poisson average of the interval heat kernels, in wall coordinates."""
import numpy as np
from numpy.polynomial.legendre import leggauss

TRUE_PARAMETER = .873


def single(d, c):
    return 3*np.cbrt(np.pi**2*d*c*c/4)


def transmission(q, c, T, order):
    # t=T^3. Remove the common leading attenuation before forming the ratio.
    d = np.array([q["D_A"],q["D_B"]])
    settings = q["coat"]
    k = np.array([np.pi if code=="DD" else .5*np.pi for code in settings])
    energy = d @ k**2
    w = np.cbrt(2*energy/c)
    nodes, weights = leggauss(order)

    def average(left, right):
        left, right = np.array(left), np.array(right)
        lo, hi = left.min(), right.max()
        # Integrate exterior wall distances in units of sqrt(T).
        cutoff = abs(hi-lo) + 16/np.sqrt(3*energy/w**4)
        grid = (nodes+1)*cutoff/2
        a,b = grid[:,None],grid[None,:]
        z = hi-lo+a+b
        ratio = 1 + z/(w*np.sqrt(T))
        # cL + energy*t/L^2 - its saddle value, factored to avoid cancellation.
        cost = energy*z**2/w**4*(2*ratio+1)/ratio**2
        kernel = np.exp(-cost)
        for i,code in enumerate(settings):
            ur = (right.max()-right[i]+b)/(w*np.sqrt(T)*ratio)
            ul = (left[i]-lo+a)/(w*np.sqrt(T)*ratio)
            total = 0.
            for mode in range(3):
                wave = k[i] + np.pi*mode
                f = np.sin(wave*ul)*np.sqrt(T) if code[0]=="D" else np.cos(wave*ul)
                g = np.sin(wave*ur)*np.sqrt(T) if code[1]=="D" else np.cos(wave*ur)
                decay = np.exp(-d[i]*(wave**2-k[i]**2)*T/(w*ratio)**2)
                total += (-1)**mode * f*g*decay
            kernel *= total/ratio
        return weights @ kernel @ weights * (cutoff/2)**2
    return average(q["left"], q["right"]) / average([0,0], [0,0])


def predict(q, c=TRUE_PARAMETER, T=1.e8, order=64):
    if q["mode"] == "single":
        return single(q["D_A"], c)
    # Remove the leading finite-time correction proportional to T^(-1/2).
    return 2*transmission(q,c,4*T,order) - transmission(q,c,T,order)
