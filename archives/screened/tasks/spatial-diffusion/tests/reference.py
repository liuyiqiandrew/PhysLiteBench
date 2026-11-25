"""Independent finite-volume discretization of isothermal diffusive flux."""
from functools import lru_cache
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix

PARAMETER = 'diffusivity'
TRUE_PARAMETER = .55
BOUNDS = (.1, 1.2)
LENGTH = 10.


def flux_operator(contrast, diffusivity, cells):
    dx = LENGTH/cells
    i = np.arange(cells)
    j = (i+1) % cells
    face = diffusivity*(1+contrast*np.cos(2*np.pi*(i+1)/cells))/dx**2
    row = np.concatenate([i, i, j, j])
    col = np.concatenate([i, j, i, j])
    data = np.concatenate([-face, face, face, -face])
    return coo_matrix((data, (row, col)), shape=(cells, cells)).tocsc()


@lru_cache(None)
def trajectory(contrast, first, second, diffusivity, end_time, cells):
    theta = 2*np.pi*(np.arange(cells)+.5)/cells
    initial = (1+first*np.cos(theta)+second*np.cos(2*theta))/cells
    matrix = flux_operator(contrast, diffusivity, cells)
    return solve_ivp(lambda t, p: matrix@p, (0., end_time), initial,
                     method='BDF', jac=matrix, rtol=2e-10, atol=1e-12,
                     dense_output=True)


def predict(experiments, diffusivity, cells=512):
    out = []
    theta = 2*np.pi*(np.arange(cells)+.5)/cells
    for e in experiments:
        if e['contrast'] == 0:
            amplitude = e['first'] if e['mode'] == 1 else e['second']
            out.append(amplitude/2*np.exp(-diffusivity*(2*np.pi*e['mode']/LENGTH)**2*e['time']))
            continue
        if e['time'] == 0:
            out.append((e['first'] if e['mode'] == 1 else e['second'])/2)
            continue
        answer = trajectory(e['contrast'], e['first'], e['second'], diffusivity, 80., cells)
        assert answer.success
        out.append(np.cos(e['mode']*theta) @ answer.sol(e['time']))
    return np.array(out)


def calibration_inputs():
    return [dict(contrast=0., first=.55, second=-.2, time=float(t), mode=m)
            for m in [1, 2] for t in np.linspace(.1, 12., 50)]


def hidden_inputs():
    return {f'contrast_{a}': [dict(contrast=a, first=b, second=c, time=float(t), mode=m)
            for m in [1, 2] for t in np.geomspace(.2, 65., 20)]
            for a, b, c in [(.7, .5, -.15), (-.6, -.3, .4), (.8, 0., 0.)]}
