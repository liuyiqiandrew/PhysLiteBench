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
    for pair in ('hh', 'cc', 'ch'):
        group = []
        for i in range(12):
            e = dict(h_lower=rng.uniform(.55, 1.65), h_upper=rng.uniform(.55, 1.65),
                     mu_lower=rng.uniform(.55, 1.6), mu_upper=rng.uniform(.55, 1.6),
                     k=rng.uniform(.7, 1.8), omega=rng.uniform(.3, 1.7),
                     temperature_lower=1., temperature_upper=1.)
            # Independent phases of the film and wall thermostats.
            amps = rng.uniform(-.25, .25, (3, 2)).tolist()
            e['grating'] = dict(k2=rng.uniform(.7, 1.8), omega2=rng.uniform(.3, 1.7),
                                angle=rng.uniform(.7, 2.8), amplitudes=amps)
            group.extend([dict(e, pair=pair, part=part) for part in ('real','imag')])
        groups['grating_'+pair] = group
    return groups


def mode_parameters(e):
    # Fourier coefficient of the temperature multiplying the stress correlator.
    if 'grating' not in e:
        return e['k'], e['omega'], 1., [1., e['temperature_lower'], e['temperature_upper']]
    record = e['grating']
    coeff = [complex(a, -b)/2 for a,b in record['amplitudes']]
    return record['k2'], record['omega2'], np.cos(record['angle']), coeff


def generator(k, mu):
    return np.array([[0, -1j*k, 1/mu, 0], [-1j*k, 0, 0, 0],
                     [4*mu*k*k, 0, 0, -1j*k], [0, 0, -1j*k, 0]])


def local_random_traction(e, side):
    from scipy.integrate import solve_ivp
    k1 = e['k']
    k2, _, angle_cos, temp = mode_parameters(e)
    mu, depth = e['mu_' + side], e['h_' + side]
    orientation = 1 if side == 'lower' else -1
    raw, dynamics = [], []
    for k in (k1, k2):
        prop = transfer(k, orientation*depth, mu)
        raw.append(prop[2:] @ np.linalg.inv(prop[:2]))
        dynamics.append(orientation*generator(k, mu))
    strength = np.diag([angle_cos/mu, 0., 2*mu*k1*k2*(1+angle_cos**2), 0.])
    wall = temp[1 if side == 'lower' else 2]
    def evolve(distance, flat):
        C = flat.reshape(4, 4)
        local = temp[0] + (wall-temp[0])*(1-distance/depth)
        return (dynamics[0] @ C + C @ dynamics[1].conj().T + local*strength).ravel()
    solution = solve_ivp(evolve, (0, depth), np.zeros(16, complex),
                         method='DOP853', rtol=2e-11, atol=1e-13)
    P,Q = [np.column_stack([-r, np.eye(2)]) for r in raw]
    return orientation*raw[0], orientation*raw[1], P @ solution.y[:, -1].reshape(4, 4) @ Q.conj().T


def noise_spectrum(e, viscosity):
    k, omega = e['k'], e['omega']
    k2, omega2, cosine, temp = mode_parameters(e)
    A, B, N = (np.zeros((4, 4), complex) for _ in range(3))
    for side in ('lower', 'upper'):
        r1, r2, noise = local_random_traction(e, side)
        A[:2, :2] += r1
        B[:2, :2] += r2
        N[:2, :2] += noise
    for matrix, wave, freq in [(A,k,omega), (B,k2,omega2)]:
        matrix[0, 0] += (viscosity+.42)*wave**2
        matrix[0, 2], matrix[1, 3] = 1j*wave*.8, 1.3*wave**2
        matrix[2, 0], matrix[2, 2] = 1j*wave, .18*wave**2 - 1j*freq
        matrix[3, 1], matrix[3, 3] = -1, -1j*freq
    # Project the shear stress, areal pressure and diffusive flux onto
    # the two probe directions using their isotropic tensor covariances.
    N[0, 0] += temp[0]*k*k2*(.42+viscosity*(2*cosine**2-1))
    N[2, 2] = temp[0]*.18*k*k2*cosine/.8
    result = np.linalg.inv(A) @ N @ np.linalg.inv(B).conj().T
    a, b = (2+'ch'.index(c) for c in e['pair'])
    value = result[a, b]
    return value.real if e['part'] == 'real' else value.imag
