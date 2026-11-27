import numpy as np


def response(e, surface_viscosity):
    k, omega = e['k'], e['omega']
    if 'pair' in e:
        return spontaneous(e, surface_viscosity)
    axis = e['axis']
    if axis == 'x':
        impedance = k**2 * (surface_viscosity + 0.42 + 0.8 / (0.18 * k**2 - 1j * omega))
    elif axis == 'y':
        impedance = k**2 * surface_viscosity
    else:
        impedance = 1.3 * k**2 / (-1j * omega)
    for side in ('lower', 'upper'):
        a = k * e['h_' + side]
        mu_k = e['mu_' + side] * k
        sh, ch = np.sinh(a), np.cosh(a)
        if axis == 'y':
            impedance += mu_k * ch / sh
        else:
            correction = a if axis == 'z' else -a
            impedance += 2 * mu_k * (sh * ch + correction) / (sh**2 - a**2)
    value = e['f' + axis] / impedance
    return value.real if e['part'] == 'real' else value.imag


def spontaneous(e, surface_viscosity):
    # Film fluctuation channels in a uniform-temperature cell.
    pair, k = e['pair'], e['k']
    if pair[0] != pair[1] or e['part'] == 'imag':
        return 0.
    is_density = pair == 'cc'
    drag = (surface_viscosity + .42)*k**2 if is_density else 0.
    for side in ('lower', 'upper'):
        a = k * e['h_' + side]
        sh, ch = np.sinh(a), np.cosh(a)
        correction = -a if is_density else a
        drag += 2*e['mu_' + side]*k*(sh*ch + correction)/(sh**2 - a**2)
    mobility = k**2/drag + .18*k**2/.8 if is_density else 1/drag
    stiffness = .8 if is_density else 1.3*k**2
    return mobility/((mobility*stiffness)**2 + e['omega']**2)


class Model:
    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        return np.array([response(e, self.surface_viscosity) for e in experiments])
