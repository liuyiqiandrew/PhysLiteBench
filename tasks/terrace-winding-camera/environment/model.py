"""Independent Gaussian bridges with a point-nucleation approximation."""
import numpy as np
from scipy.special import ndtr, roots_legendre


def image(q,rate):
    if q.get('open_pair') is not None:
        # This approximation treats a slit as a point at every small opening.
        return 0.
    bottom,top=q['bottom'],q['top']
    n=len(bottom);L=q['circumference'];a,b=q['levels']
    variation=2*rate*q['height'];sigma=np.sqrt(variation*a*(1-a))
    lo,hi=q['window'];cl,ch=q['count_window']
    nodes,weights=roots_legendre(24)
    shifts=L*np.arange(-5,6)
    def band(mu,sd,low,high):
        x=mu[...,None] if np.ndim(mu) else mu
        return np.sum(ndtr((high+shifts-x)/sd)-ndtr((low+shifts-x)/sd),axis=-1)
    poly=np.ones(1)
    for i,x0 in enumerate(bottom):
        winding,j=divmod(i+q['winding'],n)
        end=top[j]+winding*L
        mean1=x0+a*(end-x0);mean2=x0+b*(end-x0)
        later=band(mean2,np.sqrt(variation*b*(1-b)),cl,ch)
        early=1.
        if i==q['tag']:
            early=band(mean1,sigma,lo,hi)
            position=(lo+hi+(hi-lo)*nodes[:,None])/2+shifts
            density=np.exp(-.5*((position-mean1)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
            conditional=mean2+(1-b)/(1-a)*(position-mean1)
            inside=band(conditional,np.sqrt(variation*(b-a)*(1-b)/(1-a)),cl,ch)
            later=(hi-lo)/2*np.sum(weights[:,None]*density*inside)
        poly=np.convolve(poly,[early-later,later])
    return float(poly[q['count']])


class Model:
    def fit(self, records):
        self.rate = float(np.mean([row["value"]/(2*row["input"]["height"]) for row in records]))
        return self

    def predict(self, inputs):
        return np.array([2*self.rate*q["height"] if q["mode"]=="wandering" else image(q,self.rate)
                         for q in inputs],dtype=float)
