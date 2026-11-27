import numpy as np
from scipy.optimize import minimize_scalar


def response(e, surface_viscosity):
    axis, k, omega = e['axis'], e['k'], e['omega']
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


class Model:
    def fit(self, records):
        def cost(viscosity):
            return sum(((response(r['input'], viscosity) - r['value']) / r['sigma'])**2
                       for r in records)
        self.surface_viscosity = minimize_scalar(cost, bounds=(0.75, 1.45), method='bounded').x
        return self

    def predict(self, experiments):
        return np.array([response(e, self.surface_viscosity) for e in experiments])
