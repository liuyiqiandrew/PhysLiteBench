"""Independent macroscopic field/node equilibrium and electrode-charge response."""
from functools import lru_cache
import numpy as np
from scipy.optimize import root

TRUE_PARAMETER = 1.06


def experiment(stretches, shears, drive, load=None):
    f = np.diag(stretches).astype(float)
    f[0, 1], f[0, 2], f[1, 2] = shears
    e = {'deformation':f.tolist(), 'drive':np.asarray(drive,dtype=float).tolist()}
    if load is not None:
        e['load'] = float(load)
    return e


def calibration_inputs():
    return [experiment([1.,1.,stretch],[0.,0.,0.],np.diag([0.,0.,sign]))
            for stretch in np.linspace(.88,1.12,9) for sign in [-1.,1.]]


def hidden_inputs():
    groups = {name:[experiment([.96,1.04,z],[.04,-.03,.02],np.diag([.15,.05,1.]),load)
                    for z in [.9,.97,1.04,1.1]]
              for name,load in [('load_low',.5),('load_middle',1.),('load_high',2.)]}
    groups['short_anchor'] = [experiment([1.,1.,z],[0.,0.,0.],np.diag([0.,0.,sign]))
                              for z in [.91,1.03] for sign in [-1.,1.]]
    return groups


def field_reference(f, stiffness, load, electrical_coupling=1.):
    """Independent unknowns: offset, lab normal E, voltage, two face charges.

    E uses q0/(epsilon*a0^2); V uses q0*N/(epsilon*a0).
    No reduced capacitor divider, reduced energy, or forward tangent is used.
    The reference is the uniform macroscopic Maxwell problem; microscopic
    zero-field electrostatics are already part of the supplied cell energy.
    """
    material_normal = np.linalg.solve(f.T, np.array([0., 0., 1.]))
    b = np.linalg.norm(material_normal)
    normal = material_normal/b
    area = np.linalg.det(f)*b
    gap = 1/b
    e = (f.T@f-np.eye(3))/2
    h = np.array([.60*e[0, 2]+.20*e[0, 1],
                  .50*e[1, 2]-.10*e[0, 1],
                  .45*e[0, 0]+.30*e[1, 1]+.70*e[2, 2]+.20*e[0, 1]])

    def equations(y):
        s, electric, voltage, crystal_charge, load_charge = y[:3], *y[3:]
        x = s-np.array([.08, .06, .22])
        internal = stiffness*x+4*np.dot(x, x)*x-h
        internal += electrical_coupling*electric*(f.T@normal)
        return np.r_[internal,
                     crystal_charge-s[2]+area*electric,
                     voltage+gap*electric,
                     voltage if load is None else load_charge-load*voltage,
                     crystal_charge+load_charge]

    guess = np.r_[np.array([.08, .06, .22])+h/stiffness, 0., 0., .22, -.22]
    result = root(equations, guess, tol=1e-11)
    residual = float(np.max(np.abs(equations(result.x))))
    if residual > 3e-11:
        raise RuntimeError(('Field/node reference failed', result.message, residual))
    return result.x, residual


def scalar_response(f, drive, stiffness, load, step=2e-4, electrical_coupling=1.):
    charges = [field_reference(f+i*step*drive, stiffness, load, electrical_coupling)[0][5]
               for i in (-2, -1, 1, 2)]
    return float((charges[0]-8*charges[1]+8*charges[2]-charges[3])/(12*step))


@lru_cache(maxsize=16384)
def response(stiffness, f_values, g_values, load, step=2e-4):
    return scalar_response(np.asarray(f_values).reshape(3,3),
                           np.asarray(g_values).reshape(3,3),stiffness,load,step)


def predict(experiments, stiffness, step=2e-4):
    return np.asarray([response(float(stiffness),tuple(np.asarray(e['deformation'],dtype=float).ravel()),
                                tuple(np.asarray(e['drive'],dtype=float).ravel()),
                                None if e.get('load') is None else float(e['load']),step)
                       for e in experiments])
