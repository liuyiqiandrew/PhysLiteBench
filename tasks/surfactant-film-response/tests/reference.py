"""Solve the two bulk boundary-value problems with unknown wall tractions."""
import numpy as np
from scipy.linalg import expm


TRUE_PARAMETER = 1.137


def transfer(k, depth, mu):
    # State (vx, vz, stress_xz, stress_zz), derivatives in the global +z direction.
    a = np.array([[0, -1j*k, 1/mu, 0],
                  [-1j*k, 0, 0, 0],
                  [4*mu*k*k, 0, 0, -1j*k],
                  [0, 0, -1j*k, 0]])
    return expm(depth * a)[:, 2:]


def component(e, viscosity):
    k, frequency = e['k'], -1j*e['omega']
    if e['axis'] == 'y':
        # A unit transverse wall stress gives this interface velocity and stress.
        resistance = k**2 * viscosity
        for side in ('lower', 'upper'):
            mu = e['mu_' + side]
            shear = expm(e['h_' + side] * np.array([[0, 1/mu], [mu*k*k, 0]]))[:, 1]
            resistance += shear[1] / shear[0]
        z = e['fy'] / resistance
    else:
        lower = transfer(k, e['h_lower'], e['mu_lower'])
        upper = transfer(k, -e['h_upper'], e['mu_upper'])
        # Unknowns: two stresses at each plate, common vx and vz, height, density change.
        a = np.zeros((8, 8), dtype=complex)
        a[:2, :2], a[2:4, 2:4] = lower[:2], upper[:2]
        a[:2, 4:6] = a[2:4, 4:6] = -np.eye(2)
        a[4:6, :2], a[4:6, 2:4] = lower[2:], -upper[2:]
        a[4, 4], a[4, 7] = (viscosity + 0.42)*k*k, 1j*k*0.8
        a[5, 6] = 1.3*k*k
        a[6, 5], a[6, 6] = -1, frequency
        a[7, 4], a[7, 7] = 1j*k, frequency + 0.18*k*k
        b = [0, 0, 0, 0, e['fx'], e['fz'], 0, 0]
        unknown = np.linalg.solve(a, b)
        z = unknown[4 if e['axis'] == 'x' else 5]
    return z.real if e['part'] == 'real' else z.imag


def predict(experiments, viscosity):
    return np.array([component(e, viscosity) for e in experiments])


def hidden_inputs():
    groups = {'normal_drive': [], 'lateral_drive': [], 'mixed_drive': []}
    rng = np.random.default_rng(31879)
    for i in range(12):
        shallow, deep = rng.uniform(0.45, 0.7), rng.uniform(1.4, 1.9)
        lower, upper = (deep, shallow) if i % 2 else (shallow, deep)
        e = dict(h_lower=lower, h_upper=upper, mu_lower=rng.uniform(.45, 1.7),
                 mu_upper=rng.uniform(.45, 1.7), k=rng.uniform(.65, 2.1),
                 omega=rng.uniform(.25, 4.5), fy=0.0)
        for part in ('real', 'imag'):
            groups['normal_drive'].append(dict(e, fx=0., fz=1., axis='x', part=part))
            groups['lateral_drive'].append(dict(e, fx=1., fz=0., axis='z', part=part))
            for axis in ('x', 'z'):
                groups['mixed_drive'].append(dict(e, fx=.8, fz=-.9, axis=axis, part=part))
    return groups
