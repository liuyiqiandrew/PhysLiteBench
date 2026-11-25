"""Independent spatial jump process and displacement-counting moments."""
from functools import lru_cache
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

TRUE_PARAMETER=.93


def experiment(a,b,c,d,angle):
    return dict(a=float(a),b=float(b),c=float(c),d=float(d),angle=float(angle))


def calibration_inputs():
    rows=[]
    for axis in [0,1]:
        for index,amplitude in enumerate([.6,1.2,2.,3.]):
            harmonic=(-1)**index*.25*amplitude
            for angle in [.15,.6,1.0]:
                rows.append(experiment(amplitude if axis==0 else 0.,amplitude if axis==1 else 0.,harmonic if axis==0 else 0.,harmonic if axis==1 else 0.,angle if axis==0 else np.pi/2-angle))
    return rows


def hidden_inputs():
    return {
        'cellular': [experiment(5,5,0,0,.2),experiment(6,6,0,0,.8),experiment(8,8,0,0,1.3)],
        'harmonics': [experiment(5,5,.8,-.8,.35),experiment(6,6,-.8,.8,1.05),experiment(8,8,.75,-.65,2.)],
        'signed_flows': [experiment(5,-5,.6,.4,.3),experiment(-6,6,-.5,.8,.95),experiment(-8,-8,.8,.8,2.4)],
        'anchors': [experiment(0,0,0,0,.6),experiment(3,0,.75,0,.5),experiment(0,-2,0,.8,2.),experiment(.1,.1,.03,-.02,.3)]
    }


def generator(a,b,c,d,diffusivity,n):
    h=2*np.pi/n;ids=np.arange(n*n).reshape(n,n);x,y=np.indices((n,n))*h
    velocity=np.array([a*np.sin(y)+c*np.sin(2*y),b*np.sin(x)+d*np.sin(2*x)]).reshape(2,-1).T
    rows=[];cols=[];rates=[];marks=[]
    for axis in range(2):
        for sign in [-1,1]:
            rows.append(ids.ravel());cols.append(np.roll(ids,-sign,axis=axis).ravel())
            rates.append(diffusivity/h**2+sign*velocity[:,axis]/(2*h))
            mark=np.zeros((n*n,2));mark[:,axis]=sign*h;marks.append(mark)
    rows=np.concatenate(rows);cols=np.concatenate(cols);rates=np.concatenate(rates);marks=np.concatenate(marks)
    assert rates.min()>0
    size=n*n;off=sparse.coo_matrix((rates,(rows,cols)),shape=(size,size)).tocsc()
    L=off-sparse.diags(np.asarray(off.sum(axis=1)).ravel())
    first=[sparse.coo_matrix((rates*marks[:,i],(rows,cols)),shape=L.shape).tocsc() for i in range(2)]
    mean=np.column_stack([np.asarray(q.sum(axis=1)).ravel() for q in first])
    second=marks.T@(rates[:,None]*marks)/size
    return L,first,mean,second,float(rates.min())


@lru_cache(maxsize=512)
def tensor_on_grid(a,b,c,d,diffusivity,n):
    L,first,mean,second,_=generator(a,b,c,d,diffusivity,n)
    corrector=np.zeros_like(mean)
    corrector[1:]=spsolve(-L[1:,1:],mean[1:])
    corrector-=corrector.mean(axis=0)
    term=np.array([np.mean(q@corrector,axis=0) for q in first])
    return (second+term+term.T)/2


@lru_cache(maxsize=512)
def diffusion_tensor(a,b,c,d,diffusivity,coarse=48):
    low=tensor_on_grid(a,b,c,d,diffusivity,coarse)
    high=tensor_on_grid(a,b,c,d,diffusivity,2*coarse)
    return (4*high-low)/3


def predict(experiments,diffusivity,coarse=48):
    result=[]
    for e in experiments:
        K=diffusion_tensor(e['a'],e['b'],e['c'],e['d'],diffusivity,coarse)
        direction=np.array([np.cos(e['angle']),np.sin(e['angle'])])
        result.append(direction@K@direction)
    return np.asarray(result)
