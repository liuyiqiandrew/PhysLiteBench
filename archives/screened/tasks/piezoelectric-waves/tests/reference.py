"""Coupled mechanical/electrostatic generalized eigenproblem, without field elimination."""
import numpy as np
from scipy.linalg import eigvals

PARAMETER='lame_parameter'
TRUE_PARAMETER=32.


def modes(direction,lame_parameter):
    n=np.asarray(direction,dtype=float)
    # Assemble stresses and electric displacement from unit displacement and
    # potential amplitudes. The potential scale balances the saddle matrix.
    potential_scale=np.sqrt(1e9/1e-8)
    def response(displacement,potential):
        strain=(np.outer(n,displacement)+np.outer(displacement,n))/2
        field=-potential_scale*n*potential
        polarization=np.array([24*strain[0,2],24*strain[1,2],
                               -7*(strain[0,0]+strain[1,1])+18*strain[2,2]])
        electric_displacement=np.array([8e-9,8e-9,1e-8])*field+polarization
        stress=lame_parameter*1e9*np.trace(strain)*np.eye(3)+40e9*strain
        stress[0,2]-=12*field[0];stress[2,0]-=12*field[0]
        stress[1,2]-=12*field[1];stress[2,1]-=12*field[1]
        stress[0,0]+=7*field[2];stress[1,1]+=7*field[2]
        stress[2,2]-=18*field[2]
        return np.r_[stress@n/1e9,potential_scale*(n@electric_displacement)/1e9]
    matrix=np.column_stack([response(np.eye(3)[j],0.) for j in range(3)]+[response(np.zeros(3),1.)])
    mass=np.diag([1.,1.,1.,0.])
    values=eigvals(matrix,mass)
    values=values[np.isfinite(values)]
    assert len(values)==3 and np.max(abs(values.imag))<1e-8
    return np.sqrt(np.sort(values.real)*1e9/6000.)


def predict(experiments,lame_parameter=TRUE_PARAMETER):
    return np.array([modes(e['direction'],lame_parameter)[e['branch']] for e in experiments])


def reading(theta,azimuth=0.,branch=2):
    return dict(direction=[float(np.sin(theta)*np.cos(azimuth)),float(np.sin(theta)*np.sin(azimuth)),float(np.cos(theta))],branch=int(branch))


def calibration_inputs():
    return [reading(theta) for theta in [0.,np.pi] for _ in range(24)]+[
        reading(np.pi/2,azimuth,branch) for branch in [0,1] for azimuth in np.linspace(0,2*np.pi,24,endpoint=False)]


def hidden_inputs():
    return {'lower_shear':[reading(theta,.3,0) for theta in np.linspace(.25,.7,20)],
            'upper_shear':[reading(theta,1.1,1) for theta in np.linspace(.2,.8,20)],
            'oblique_branches':[reading(theta,azimuth,branch) for theta,azimuth in zip(np.linspace(.6,1.05,12),np.linspace(0,2*np.pi,12)) for branch in [0,1]]}
