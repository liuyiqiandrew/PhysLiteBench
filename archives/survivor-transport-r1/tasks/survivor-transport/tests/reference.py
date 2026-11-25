"""Conservative annular transport and axial counting-field reference."""
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh_tridiagonal

TRUE_PARAMETER = 1.05


def operator(diffusivity, radius, capture, cells):
    dr = radius/cells
    edges = np.linspace(0,radius,cells+1)
    r = (edges[:-1]+edges[1:])/2
    volume = (edges[1:]**2-edges[:-1]**2)/2
    conductance = diffusivity*edges[1:-1]/dr
    diagonal = np.zeros(cells)
    diagonal[:-1] += conductance
    diagonal[1:] += conductance
    diagonal[-1] += radius*capture/(1+capture*dr/(2*diffusivity))
    return r,volume,-diagonal/volume,conductance/np.sqrt(volume[:-1]*volume[1:])


def principal(diagonal, off):
    n = len(diagonal)
    return float(eigh_tridiagonal(diagonal,off,eigvals_only=True,
        select='i',select_range=(n-1,n-1))[0])


def discrete_readout(diffusivity,radius,capture,plug,peak,cells,counting_step=.04):
    r,volume,diagonal,off = operator(diffusivity,radius,capture,cells)
    u = plug+peak*(1-r*r/radius**2)
    base = principal(diagonal,off)
    def derivative(h):
        positive = principal(diagonal+h*u+diffusivity*h*h,off)
        negative = principal(diagonal-h*u+diffusivity*h*h,off)
        return (positive-negative)/(2*h)
    coarse = derivative(2*counting_step)
    fine = derivative(counting_step)
    return np.array([-base,(4*fine-coarse)/3])


@lru_cache(maxsize=2048)
def response(diffusivity,radius,capture,plug,peak,cells=256):
    coarse = discrete_readout(diffusivity,radius,capture,plug,peak,cells)
    fine = discrete_readout(diffusivity,radius,capture,plug,peak,2*cells)
    return (4*fine-coarse)/3


def predict(experiments,diffusivity):
    result = []
    for e in experiments:
        r = response(diffusivity,e['radius'],e['capture'],e['plug'],e['peak'])
        result.append(r[0 if e['observable']=='loss_rate' else 1])
    return np.array(result,dtype=float)


def experiment(radius,capture,plug=0.,peak=0.,observable='drift'):
    return dict(radius=radius,capture=capture,plug=plug,peak=peak,observable=observable)


def hidden_inputs():
    return {
        'capture_scan':[experiment(1.,k,peak=1.) for k in [6.,12.,24.]],
        'geometry_scan':[experiment(r,16.,peak=.8) for r in [.75,1.,1.25]],
        'reverse_flow':[experiment(.8,8.,peak=-1.2),experiment(1.2,20.,peak=-.7),
                        experiment(1.3,24.,peak=-1.4)],
        'shared_anchors':[experiment(.95,0.,peak=1.1),experiment(1.15,7.,plug=-.8),
                          experiment(.85,4.,peak=.9,observable='loss_rate')],
    }
