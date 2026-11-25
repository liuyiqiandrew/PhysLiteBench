"""Conservative finite-volume OU generator and conditional Bloch evolution."""
import json
import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

PARAMETER = 'noise_width'
TRUE_PARAMETER = .95
CORRELATION_TIME = 1.


def calibration_inputs():
    return [dict(preparation=[np.pi/2,np.pi/2],readout=[np.pi/2,-np.pi/2],
                 segments=[dict(duration=float(t),field=[0.,0.,0.])])
            for t in np.linspace(.03,3.,100)]


def hidden_inputs():
    return {
        'rabi_drive': [dict(preparation=[0.,0.],readout=[0.,0.],
                           segments=[dict(duration=float(t),field=[3.,0.,0.])])
                       for t in np.linspace(.5,4.,24)],
        'changed_axis': [dict(preparation=[np.pi/2,1.2],readout=[.4,-1.3],
                             segments=[dict(duration=1.,field=[2.5,0.,.3]),
                                       dict(duration=float(t),field=[0.,2.8,-.4])])
                         for t in np.linspace(.3,3.,24)],
        'tilted_drive': [dict(preparation=[.8,1.4],readout=[-1.1,1.2],
                             segments=[dict(duration=float(t),field=[1.5,2.,.8])])
                         for t in np.linspace(.5,4.,24)]}


def cross(field):
    x,y,z = field
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def rotate(vector,specification):
    azimuth,angle = specification
    return Rotation.from_rotvec(angle*np.array([np.cos(azimuth),np.sin(azimuth),0.])).apply(vector)


def predict(experiments, noise_width, cells=321):
    y = np.linspace(-8.,8.,cells)
    spacing = y[1]-y[0]
    potential = y*y/2
    right = np.exp(-np.diff(potential)/2)/(CORRELATION_TIME*spacing**2)
    left = np.exp(np.diff(potential)/2)/(CORRELATION_TIME*spacing**2)
    diagonal = -np.r_[right,0.]-np.r_[0.,left]
    ou = sparse.diags([right,diagonal,left],[-1,0,1],shape=(cells,cells),format='csc')
    diffusion = sparse.kron(ou,np.eye(3),format='csc')
    field_noise = sparse.kron(sparse.diags(noise_width*y),cross([0.,0.,1.]),format='csc')
    mass = np.exp(-potential); mass /= mass.sum()
    out = np.empty(len(experiments))
    groups = {}
    for j,e in enumerate(experiments):
        if all(np.linalg.norm(s['field'])==0 for s in e['segments']):
            t = sum(s['duration'] for s in e['segments'])
            v = rotate(np.array([0.,0.,1.]),e['preparation'])
            v[:2] *= np.exp(-noise_width**2*CORRELATION_TIME*(t-CORRELATION_TIME*(-np.expm1(-t/CORRELATION_TIME))))
            out[j] = (1+rotate(v,e['readout'])[2])/2
            continue
        key = json.dumps(dict(e,segments=e['segments'][:-1]+[dict(e['segments'][-1],duration=0.)]),sort_keys=True)
        groups.setdefault(key,[]).append((j,e))
    for group in groups.values():
        example = group[0][1]
        initial = mass[:,None]*rotate(np.array([0.,0.,1.]),example['preparation'])[None,:]
        state = initial.ravel()
        def evolution(state,field,duration,dense=False):
            matrix = diffusion+field_noise+sparse.kron(sparse.eye(cells),cross(field),format='csc')
            if duration == 0:
                return (lambda t: state) if dense else state
            sol = solve_ivp(lambda t,v: matrix@v,(0.,duration),state,jac=matrix,
                            method='BDF',rtol=2e-10,atol=2e-13,dense_output=dense)
            assert sol.success
            return sol.sol if dense else sol.y[:,-1]
        for segment in example['segments'][:-1]:
            state = evolution(state,segment['field'],segment['duration'])
        last = example['segments'][-1]
        solution = evolution(state,last['field'],max(e['segments'][-1]['duration'] for _,e in group),dense=True)
        for j,e in group:
            v = solution(e['segments'][-1]['duration']).reshape(cells,3).sum(axis=0)
            out[j] = (1+rotate(v,e['readout'])[2])/2
    return out
