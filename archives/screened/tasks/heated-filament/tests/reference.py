"""Spatial finite-difference adjoint response, independent of modal-noise projection."""
import numpy as np
from scipy.linalg import solve_banded

PARAMETER = 'friction'
TRUE_PARAMETER = 1.1


def pair(first, second, sign=1):
    weights = np.zeros(6)
    weights[first-1] = first**2
    weights[second-1] = sign*second**2
    return (weights/np.linalg.norm(weights)).tolist()


def experiment(weights, frequency=0., temperature=.02, contrast=0., wavenumber=1, phase=0.):
    return dict(weights=list(weights),frequency=float(frequency),temperature=float(temperature),contrast=float(contrast),wavenumber=int(wavenumber),phase=float(phase))


def calibration_inputs():
    sensors = [pair(1,2),pair(1,3,-1),pair(2,3),pair(1,6,-1)]
    return [experiment(w,frequency=f,temperature=t) for w in sensors for f in [0.,.12,.3,.45] for t in [.01,.025]]


def hidden_inputs():
    anchors=[]
    for mode in range(1,7):
        w=np.eye(6)[mode-1].tolist()
        for phase in [-.7,.9]:
            anchors.append(experiment(w,frequency=1.5*mode,temperature=.03,contrast=.8,wavenumber=mode,phase=phase))
    return {
        'single_mode_anchors':anchors,
        'mixed_sensors':[experiment(pair(m,n,s),frequency=f,temperature=.02,contrast=.8,wavenumber=n-m) for m,n in [(1,2),(1,3),(2,3),(3,4)] for s in [-1,1] for f in [0.,.5,2.]],
        'spatial_phase':[experiment(pair(1,2,s),frequency=f,temperature=.025,contrast=.85,wavenumber=1,phase=p) for s in [-1,1] for p in [.3,.9,2.2,2.8] for f in [0.,.8,2.]],
        'frequency_spectrum':[experiment(pair(1,2,s),frequency=f,temperature=.015,contrast=.85,wavenumber=1) for s in [-1,1] for f in [0.,.3,.8,1.5,3.,5.,8.,12.,20.]],
    }


def spatial_spectrum(experiment, friction, intervals=512):
    dx=np.pi/intervals
    x=np.arange(1,intervals)*dx
    w=np.sqrt(2/np.pi)*sum(a*np.sin((n+1)*x) for n,a in enumerate(experiment['weights']))
    matrix=np.zeros((3,len(x)),dtype=complex)
    matrix[0,1:]=-1/dx**2
    matrix[1]=2/dx**2+1j*friction*experiment['frequency']
    matrix[2,:-1]=-1/dx**2
    response=solve_banded((1,1),matrix,w)
    temperature=experiment['temperature']*(1+experiment['contrast']*np.cos(experiment['wavenumber']*x+experiment['phase']))
    return float(2*friction*dx*np.sum(temperature*abs(response)**2))


def predict(experiments, friction, intervals=512):
    result=[]
    for e in experiments:
        fine=spatial_spectrum(e,friction,intervals)
        coarse=spatial_spectrum(e,friction,intervals//2)
        result.append((4*fine-coarse)/3)
    return np.array(result)
