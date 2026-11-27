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
    return np.array([noise_spectrum(e, viscosity) if 'pair' in e else component(e, viscosity)
                     for e in experiments])


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
    spectra = {'height_power': ('hh', 'real'), 'density_power': ('cc', 'real'),
               'cross_lag': ('ch', 'imag'), 'cross_in_phase': ('ch', 'real')}
    for name, (pair, part) in spectra.items():
        groups[name] = []
        for i in range(16):
            e = dict(h_lower=rng.uniform(.5, 1.6), h_upper=rng.uniform(.5, 1.6),
                     mu_lower=rng.uniform(.55, 1.6), mu_upper=rng.uniform(.55, 1.6),
                     k=rng.uniform(.7, 1.9), omega=rng.uniform(.25, 2.4))
            if i % 4 == 0:
                e.update(h_upper=e['h_lower'], mu_upper=e['mu_lower'])
            lo, hi = rng.uniform(.45, .62), rng.uniform(1.55, 1.8)
            if i % 4 == 1:
                lo = hi
            if i % 2:
                lo, hi = hi, lo
            groups[name].append(dict(e, temperature_lower=lo, temperature_upper=hi,
                                     pair=pair, part=part))
    return groups


def local_random_traction(k, e, side):
    from scipy.integrate import solve_ivp
    mu, depth = e['mu_' + side], e['h_' + side]
    orientation = 1 if side == 'lower' else -1
    prop = transfer(k, orientation*depth, mu)
    raw = prop[2:] @ np.linalg.inv(prop[:2])
    # Propagate random bulk stresses from wall to film in the total-stress
    # state formulation. Wall tractions are free to impose the end velocity.
    generator = orientation * np.array([[0, -1j*k, 1/mu, 0],
                                        [-1j*k, 0, 0, 0],
                                        [4*mu*k*k, 0, 0, -1j*k],
                                        [0, 0, -1j*k, 0]])
    strength = np.diag([1/mu, 0., 4*mu*k*k, 0.])
    def evolve(distance, flat):
        C = flat.reshape(4, 4)
        temp = 1 + (e['temperature_' + side] - 1) * (1-distance/depth)
        return (generator @ C + C @ generator.conj().T + temp*strength).ravel()
    solution = solve_ivp(evolve, (0, depth), np.zeros(16, complex),
                         method='DOP853', rtol=2e-11, atol=1e-13)
    C = solution.y[:, -1].reshape(4, 4)
    project = np.column_stack([-raw, np.eye(2)])
    return orientation*raw, project @ C @ project.conj().T


def noise_spectrum(e, viscosity):
    k, omega = e['k'], e['omega']
    A = np.zeros((4, 4), complex)  # vx, vz, c, height
    N = np.zeros_like(A)
    for side in ('lower', 'upper'):
        resistance, random_stress = local_random_traction(k, e, side)
        A[:2, :2] += resistance
        N[:2, :2] += random_stress
    A[0, 0] += (viscosity+.42)*k*k
    N[0, 0] += (viscosity+.42)*k*k
    A[0, 2], A[1, 3] = 1j*k*.8, 1.3*k*k
    A[2, 0], A[2, 2] = 1j*k, .18*k*k - 1j*omega
    A[3, 1], A[3, 3] = -1, -1j*omega
    N[2, 2] = .18*k*k/.8
    inv = np.linalg.inv(A)
    result = inv @ N @ inv.conj().T
    a, b = (2+'ch'.index(c) for c in e['pair'])
    value = result[a, b]
    return value.real if e['part'] == 'real' else value.imag
