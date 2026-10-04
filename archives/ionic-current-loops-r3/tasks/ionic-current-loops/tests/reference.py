"""Conservative face fluxes and an independent periodic scalar-potential solve."""
from functools import lru_cache
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, cg

TRUE_PARAMETER = 1.13
PARAMETER = 'diffusivity'


@lru_cache(128)
def fields(amplitudes, phase, points):
    h = 2*np.pi/points
    coordinates = np.arange(points)*h
    x, y = np.meshgrid(coordinates, coordinates, indexing='ij')
    a, b, c, d = amplitudes
    first = 1+a*np.cos(x)+b*np.cos(y)
    second = 1+c*np.cos(x+phase)+d*np.cos(x+y)
    concentration = np.array([first, second, first+second])
    ratios = np.array([1.,4.,1.]); charges = np.array([1.,1.,-1.])
    sigma = np.einsum('i,ijk->jk', ratios, concentration)
    g = np.einsum('i,ijk->jk', ratios*charges, concentration)
    face_sigma = np.array([.5*(sigma+np.roll(sigma,-1,axis)) for axis in range(2)])
    ids = np.arange(points*points).reshape(points,points)
    rows, columns, data = [], [], []
    for axis in range(2):
        left = ids.ravel(); right = np.roll(ids,-1,axis).ravel()
        conductance = face_sigma[axis].ravel()/h**2
        rows.extend([left,left,right,right]); columns.extend([left,right,left,right])
        data.extend([conductance,-conductance,-conductance,conductance])
    matrix = coo_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(columns))),
                        shape=(points*points,points*points)).tocsr()
    operator = LinearOperator(matrix.shape,matvec=lambda v:matrix@v+v.mean(),dtype=float)
    k = np.fft.fftfreq(points,1/points)
    kx, ky = np.meshgrid(k,k,indexing='ij')
    denominator = sigma.mean()*4/h**2*(np.sin(np.pi*kx/points)**2+np.sin(np.pi*ky/points)**2)
    denominator[0,0] = 1.
    def precondition(v):
        return np.fft.ifft2(np.fft.fft2(v.reshape(points,points))/denominator).real.ravel()
    pre = LinearOperator(matrix.shape,matvec=precondition,dtype=float)
    def gradient(value):
        return np.array([(np.roll(value,-1,axis)-value)/h for axis in range(2)])
    def divergence(value):
        return sum((value[axis]-np.roll(value[axis],1,axis))/h for axis in range(2))
    potential,status = cg(operator,divergence(gradient(g)).ravel(),M=pre,
                          rtol=1e-12,atol=1e-14,maxiter=1000)
    if status != 0:
        raise RuntimeError('Finite-volume potential failed.')
    electric = -gradient(potential.reshape(points,points))
    flux = []
    for i in range(3):
        faces = np.array([.5*(concentration[i]+np.roll(concentration[i],-1,axis)) for axis in range(2)])
        flux.append(-ratios[i]*gradient(concentration[i])+charges[i]*ratios[i]*faces*electric)
    return x,y,np.array(flux)


def prediction_at_resolution(e, points):
    x,y,flux = fields(tuple(e['amplitudes']),e['phase'],points)
    m,n = e['detector']; test = np.cos(m*x+n*y+np.pi/4); h=2*np.pi/points
    gradient = np.array([(np.roll(test,-1,axis)-test)/h for axis in range(2)])
    return 2*np.mean(np.sum(flux[e['species']]*gradient,axis=0))


def predict(experiments, diffusivity=TRUE_PARAMETER, points=64):
    return np.array([(4*prediction_at_resolution(e,2*points)-prediction_at_resolution(e,points))/3
                     for e in experiments])*diffusivity


def experiment(amplitudes=(.6,.25,.7,.2),phase=1.4,species=0,detector=(1,0)):
    return dict(amplitudes=[float(x) for x in amplitudes],phase=float(phase),species=int(species),detector=list(detector))


def calibration_inputs():
    settings=[(.55,.25,.65,.2),(.6,.25,.7,.2),(.65,.2,.7,.2),(.55,.3,.65,.25)]
    unique=[experiment(a,0,species,mode) for a in settings for species in [0,1]
            for mode in [(1,0),(0,1),(1,1),(2,1)]]
    return [dict(e) for _ in range(9) for e in unique]


def hidden_inputs():
    return {
        'phase_sweep':[experiment(phase=q) for q in np.linspace(1.,1.8,9)],
        'composition_sweep':[experiment((a,.25,c,.2),phase=1.4)
                             for a,c in zip(np.linspace(.5,.65,9),np.linspace(.6,.7,9))],
        'mixed_profiles':[experiment((.6,b,.7,d),phase=q)
                          for b,d,q in zip(np.linspace(.2,.3,9),np.linspace(.15,.25,9),np.linspace(1.05,1.85,9))],
        'symmetric_anchors':[experiment((.55,.25,.65,.2),q,species,mode)
                              for q in [0.,np.pi] for species,mode in [(0,(1,0)),(1,(0,1)),(2,(1,1))]]
    }
