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


def spectrum(e, viscosity):
    k, omega = e['k'], e['omega']
    # Deterministic hydrodynamic resistance, with static surface viscous stress.
    R = bulk_drag(e)[np.ix_([0, 2], [0, 2])]
    surface_drag = (viscosity + 0.42) * k**2
    R[0, 0] += surface_drag
    # q=(fractional surfactant density, height), q_dot=B v before diffusion.
    B = np.diag([-1j*k, 1.])
    elastic = np.diag([0.8, 1.3*k**2])
    transport = B @ np.linalg.inv(R)
    diffusion = np.diag([0.18*k**2/0.8, 0.])
    mobility = transport @ B.conj().T + diffusion
    noise = mobility
    green = np.linalg.inv(mobility @ elastic - 1j*omega*np.eye(2))
    psd = green @ noise @ green.conj().T
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
