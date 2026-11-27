import numpy as np
from scipy.optimize import minimize_scalar


def bulk_drag(e):
    k = e['k']
    drag = np.zeros((3, 3), dtype=complex)
    for side, orientation in [('lower', 1), ('upper', -1)]:
        a = k * e['h_' + side]
        mu_k = e['mu_' + side] * k
        sh, ch = np.sinh(a), np.cosh(a)
        factor = 2 * mu_k / (sh**2 - a**2)
        drag[0, 0] += factor * (sh * ch - a)
        drag[1, 1] += mu_k * ch / sh
        drag[2, 2] += factor * (sh * ch + a)
        drag[0, 2] -= orientation * 1j * factor * a**2
    drag[2, 0] = drag[0, 2].conjugate()
    return drag


def response(e, surface_viscosity):
    if 'pair' in e:
        return spectrum(e, surface_viscosity)
    k, omega = e['k'], e['omega']
    drag = bulk_drag(e)
    drag[0, 0] += k**2 * (surface_viscosity + 0.42 + 0.8 / (0.18 * k**2 - 1j * omega))
    drag[1, 1] += surface_viscosity * k**2
    drag[2, 2] += 1.3 * k**2 / (-1j * omega)
    velocity = np.linalg.solve(drag, [e['fx'], e['fy'], e['fz']])
    value = velocity['xyz'.index(e['axis'])]
    return value.real if e['part'] == 'real' else value.imag


def spectrum_inputs(e):
    """The grating selects the Fourier coefficient at the difference of the probes."""
    grating = e.get('grating')
    if grating is None:
        temperatures = (1., e['temperature_lower'], e['temperature_upper'])
        return e, 1., temperatures
    other = dict(e, k=grating['k2'], omega=grating['omega2'])
    temperatures = [(cosine - 1j*sine)/2 for cosine, sine in grating['amplitudes']]
    return other, 1., temperatures


def strain(k, depth, orientation, distance):
    a = k * depth
    sh, ch = np.sinh(a), np.cosh(a)
    end = [[a*ch-sh, a*sh], [a*sh, a*ch+sh]]
    coeff = np.linalg.solve(end, [[0, orientation], [-1j, 0]])
    s = k * distance
    sh, ch = np.sinh(s), np.cosh(s)
    u = 1j * (np.column_stack([s*sh, sh+s*ch]) @ coeff)
    shear = 2j*k * (np.column_stack([s*ch, ch+s*sh]) @ coeff)
    return u, shear


def bulk_noise(e, quadrature_order=16):
    other, angle_cos, temp = spectrum_inputs(e)
    k1, k2 = e['k'], other['k']
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    total = np.zeros((2, 2), complex)
    for side, orientation, wall_temp in [('lower', 1, temp[1]), ('upper', -1, temp[2])]:
        h, mu = e['h_'+side], e['mu_'+side]
        distances = h*(1+nodes)/2
        u, shear = strain(k1, h, orientation, distances)
        v, shear2 = strain(k2, h, orientation, distances)
        T = temp[0] + (wall_temp - temp[0])*(1-nodes)/2
        weighted = (weights*T*mu*h/2)[:, None]
        total += (2*k1*k2*(1+angle_cos**2) * (u.conj().T @ (weighted*v))
                  + angle_cos * (shear.conj().T @ (weighted*shear2)))
    return total


def propagation(e, viscosity):
    k = e['k']
    R = bulk_drag(e)[np.ix_([0, 2], [0, 2])]
    R[0, 0] += (viscosity + .42)*k*k
    B = np.diag([-1j*k, 1.])
    transport = B @ np.linalg.inv(R)
    mobility = transport @ B.conj().T + np.diag([.18*k*k/.8, 0.])
    green = np.linalg.inv(mobility @ np.diag([.8, 1.3*k*k]) - 1j*e['omega']*np.eye(2))
    return green, transport


def spectrum(e, viscosity):
    other, angle_cos, temperatures = spectrum_inputs(e)
    film = temperatures[0]
    kprod = e['k'] * other['k']
    green, left = propagation(e, viscosity)
    green2, right = propagation(other, viscosity)
    drag_noise = bulk_noise(e)
    drag_noise[0, 0] += film*kprod*(.42 + viscosity*(2*angle_cos**2 - 1))
    diffusion = np.diag([film*.18*kprod*angle_cos/.8, 0.])
    noise = left @ drag_noise @ right.conj().T + diffusion
    psd = green @ noise @ green2.conj().T
    a, b = ('ch'.index(s) for s in e['pair'])
    value = psd[a, b]
    return value.real if e['part'] == 'real' else value.imag


class Model:
    def fit(self, records):
        def cost(viscosity):
            return sum(((response(r['input'], viscosity) - r['value']) / r['sigma'])**2
                       for r in records)
        self.surface_viscosity = minimize_scalar(cost, bounds=(0.75, 1.45), method='bounded').x
        return self

    def predict(self, experiments):
        return np.array([response(e, self.surface_viscosity) for e in experiments])
