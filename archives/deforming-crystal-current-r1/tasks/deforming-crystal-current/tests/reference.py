"""Grounded slab electrostatics with separately relaxed internal coordinates."""
from functools import lru_cache
import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import root

TRUE_PARAMETER = 1.06


def experiment(stretches, shears, drive):
    f = np.diag(stretches).astype(float)
    f[0, 1], f[0, 2], f[1, 2] = shears
    return {'deformation':f.tolist(), 'drive':np.asarray(drive,dtype=float).tolist()}


def calibration_inputs():
    return [experiment([1.,1.,stretch],[0.,0.,0.],np.diag([0.,0.,sign]))
            for stretch in np.linspace(.88,1.12,9) for sign in [-1.,1.]]


def hidden_inputs():
    groups = {}
    for name,stretch in [('compressed',.92),('central',1.),('extended',1.08)]:
        groups[name] = [experiment([stretch,y,z],[.05,-.04,.03],g)
                       for y in [.95,1.05] for z in [.96,1.06]
                       for g in [np.diag([1.,0.,0.]),np.array([[.8,.15,0.],[0.,0.,0.],[0.,0.,.15]])]]
    groups['longitudinal_anchor'] = [experiment([1.,1.,z],[0.,0.,0.],np.diag([0.,0.,sign]))
                                    for z in [.91,1.03] for sign in [-1.,1.]]
    return groups


def equilibrium(f, stiffness):
    strain = (f.T@f-np.eye(3))/2
    field = np.array([.60*strain[0,2]+.20*strain[0,1],
                      .50*strain[1,2]-.10*strain[0,1],
                      .45*strain[0,0]+.30*strain[1,1]+.70*strain[2,2]+.20*strain[0,1]])
    result = root(lambda x: stiffness*x+4*np.dot(x,x)*x-field,field/stiffness,tol=1e-12)
    residual = np.linalg.norm(stiffness*result.x+4*np.dot(result.x,result.x)*result.x-field)
    if residual > 1e-11:
        raise RuntimeError('Internal equilibrium did not converge')
    return np.array([.08,.06,.22])+result.x


def electrode_charge(f, stiffness, cells=9, grid=2048, electrode='upper'):
    s = equilibrium(f,stiffness)
    if not 0 < .20+s[2] < 1:
        raise ValueError('Selected cell termination changed')
    spacing = cells/grid
    charge = np.zeros(grid+1)
    for j in range(cells):
        for position,value in [(j+.20,1.),(j+.20+s[2],-1.)]:
            index = position/spacing
            lower = int(np.floor(index));fraction = index-lower
            charge[lower] += value*(1-fraction)
            charge[lower+1] += value*fraction
    band = np.zeros((3,grid-1))
    band[0,1:] = -1;band[1] = 2;band[2,:-1] = -1
    potential = solve_banded((1,1),band,spacing*charge[1:-1])
    return float(-potential[-1]/spacing if electrode=='upper' else -potential[0]/spacing)


@lru_cache(maxsize=16384)
def response(stiffness, f_values, g_values, step=2e-4, cells=9, grid=2048):
    f = np.asarray(f_values).reshape(3,3)
    g = np.asarray(g_values).reshape(3,3)
    return (electrode_charge(f-2*step*g,stiffness,cells,grid)
            -8*electrode_charge(f-step*g,stiffness,cells,grid)
            +8*electrode_charge(f+step*g,stiffness,cells,grid)
            -electrode_charge(f+2*step*g,stiffness,cells,grid))/(12*step)


def predict(experiments, stiffness, step=2e-4, cells=9, grid=2048):
    return np.asarray([response(float(stiffness),tuple(np.asarray(e['deformation'],dtype=float).ravel()),
                                tuple(np.asarray(e['drive'],dtype=float).ravel()),step,cells,grid)
                       for e in experiments])
