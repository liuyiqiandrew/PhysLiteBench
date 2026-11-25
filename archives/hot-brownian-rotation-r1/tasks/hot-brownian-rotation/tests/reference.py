"""Radial angular-momentum shells driven by independent thermal stress torques."""
from functools import lru_cache
import json
import numpy as np
from scipy.linalg import solve_banded

TRUE_PARAMETER=1.07


@lru_cache(256)
def shell_response(key,viscosity,cells=512,extent=16.):
    e=json.loads(key);a=e['radius'];omega=e['omega']
    outer=33*a if omega==0 else a+max(32*a,extent*np.sqrt(2*viscosity/omega))
    faces=np.geomspace(a,outer,cells+1)
    centers=np.sqrt(faces[:-1]*faces[1:])
    nodes=np.concatenate(([a],centers,[np.inf]))
    inverse_cube=nodes**-3
    resistance_difference=inverse_cube[:-1]-inverse_cube[1:]
    conductance=8*np.pi*viscosity/resistance_difference
    temperature=e['ambient']+.75*e['rise']*a*(nodes[:-1]**-4-nodes[1:]**-4)/resistance_difference
    inertia=(8*np.pi/15)*(faces[1:]**5-faces[:-1]**5)
    diagonal=conductance[:-1]+conductance[1:]-1j*omega*inertia
    matrix=np.zeros((3,cells),complex);matrix[1]=diagonal
    matrix[0,1:]=-conductance[1:-1];matrix[2,:-1]=-conductance[1:-1]
    forcing=np.zeros(cells);forcing[0]=conductance[0]
    response=solve_banded((1,1),matrix,forcing)
    differences=np.diff(np.concatenate(([1.],response,[0.])))
    spectrum=2*np.dot(temperature*conductance,abs(differences)**2)
    impedance=conductance[0]*(1-response[0])
    dissipation=np.dot(conductance,abs(differences)**2)
    return float(spectrum),impedance,float(dissipation)


def predict(experiments,viscosity=TRUE_PARAMETER,cells=384):
    out=[]
    for e in experiments:
        key=json.dumps(e,sort_keys=True)
        coarse=shell_response(key,viscosity,cells)[0]
        fine=shell_response(key,viscosity,2*cells)[0]
        out.append((4*fine-coarse)/3)
    return np.asarray(out)


def experiment(radius=1.,ambient=1.,rise=1.5,omega=30.):
    return dict(radius=float(radius),ambient=float(ambient),rise=float(rise),omega=float(omega))


def calibration_inputs():
    unique=[experiment(radius=.8+.035*i,ambient=.8+.035*i,rise=(.8+.035*i)*(.35+.13*(i%8)),omega=0.) for i in range(12)]
    return [dict(e) for _ in range(24) for e in unique]


def hidden_inputs():
    groups={'frequency':[],'radius':[],'heating':[],'anchors':[]}
    for i in range(12):
        groups['frequency'].append(experiment(omega=12+4*i,rise=1.5+.035*i))
        groups['radius'].append(experiment(radius=.8+.035*i,omega=45+5*i,rise=1.8))
        ambient=.85+.025*i
        groups['heating'].append(experiment(ambient=ambient,rise=ambient*(1.4+.045*i),omega=90-3*i))
        groups['anchors'].append(experiment(radius=.85+.025*i,ambient=.85+.025*i,rise=0. if i%2 else 1.5,omega=3+10*i if i%2 else 0.))
    return groups
