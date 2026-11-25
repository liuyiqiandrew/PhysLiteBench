"""Average coherent spin rotations along diffusive displacements."""
import numpy as np
from numpy.polynomial.hermite import hermgauss
from functools import lru_cache

PARAMETER='diffusivity'
TRUE_PARAMETER=.12


def experiment(time,mode=1,cosine=(0.,0.,.8),sine=(.8,0.,0.),component=2,quadrature='cosine'):
    return dict(time=float(time),mode=int(mode),cosine=list(cosine),sine=list(sine),component=int(component),quadrature=quadrature)


def calibration_inputs():
    return [experiment(t,mode=0,cosine=c,sine=(0.,0.,0.),component=i) for c,i in [((.6,0.,0.),0),((0.,0.,.8),2)]
            for t in np.linspace(.2,10.,50)]


def hidden_inputs():
    return {'right_handed':[experiment(t) for t in np.linspace(.2,10.,24)],
            'left_handed':[experiment(t,sine=(-.8,0.,0.)) for t in np.linspace(.2,8.,24)],
            'standing_wave':[experiment(t,mode=2,sine=(0.,0.,0.),component=0,quadrature='sine') for t in np.linspace(.1,8.,24)]}


@lru_cache(8)
def quadrature(order):
    x,w=hermgauss(order)
    return x,w/np.sqrt(np.pi)


def predict(experiments,diffusivity,order=128):
    nodes,weights=quadrature(order);out=[]
    for e in experiments:
        displacement=np.sqrt(4*diffusivity*e['time'])*nodes
        cosine,sine=np.cos(displacement),np.sin(displacement)
        initial=np.array(e['cosine'])-1j*np.array(e['sine'])
        rotated=np.column_stack([cosine*initial[0]+sine*initial[2],np.full(len(nodes),initial[1]),-sine*initial[0]+cosine*initial[2]])
        state=np.sum(weights[:,None]*np.exp(-1j*e['mode']*displacement)[:,None]*rotated,axis=0)
        value=state[e['component']]
        out.append(float(value.real if e['quadrature']=='cosine' else -value.imag))
    return np.array(out)
