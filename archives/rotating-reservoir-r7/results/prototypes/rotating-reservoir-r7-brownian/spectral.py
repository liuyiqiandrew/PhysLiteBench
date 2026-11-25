"""Independent continuous backward-generator SCGF collocation; proposal only.

The jump quadrature evaluates the stipulated collision law directly. OU
derivatives use Chebyshev collocation rather than the positive FV discretization.
Gauge q=0 gives the literal tilted generator. Gauge q=chi cancels collision
marks. The boundary is transformed consistently from original reflecting
f'=0, or explicitly varied to gauge-Neumann only as a tail sensitivity check.
"""
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigvals, solve


def interpolation(nodes, weights, targets):
    delta = targets[:, None] - nodes[None, :]
    near = abs(delta) < 1e-13
    value = weights[None, :] / np.where(near, 1., delta)
    value /= value.sum(axis=1)[:, None]
    for i in np.flatnonzero(near.any(axis=1)):
        value[i] = 0.
        value[i, np.argmax(near[i])] = 1.
    return value


@lru_cache(2048)
def scgf(chi, ta, tb, omega=0., eta=0., flip=1., gamma=.67,
         points=121, extent=9., quadrature=140, gauge='piecewise',
         boundary='reflecting'):
    if gauge == 'piecewise':
        q = min(chi, 0.)
    elif gauge == 'full':
        q = chi
    elif gauge == 'none':
        q = 0.
    else:
        q = float(gauge)
    width = extent * np.sqrt(max(ta, tb)) + abs(omega)
    x = width * np.cos(np.arange(points) * np.pi / (points - 1))
    barycentric = (-1.) ** np.arange(points)
    barycentric[[0, -1]] *= .5
    delta = x[:, None] - x[None, :]
    derivative = (barycentric[None, :] / barycentric[:, None]) / np.where(
        np.eye(points, dtype=bool), 1., delta)
    np.fill_diagonal(derivative, 0.)
    np.fill_diagonal(derivative, -derivative.sum(axis=1))
    second = derivative @ derivative
    nodes, weights = leggauss(quadrature)
    size = points - 2
    extensions = []
    for sign in (1., -1.):
        u = sign * omega
        extension = np.zeros((points, size))
        extension[1:-1] = np.eye(size)
        rows = derivative[[0, -1]].copy()
        if boundary == 'reflecting':
            # f_old = exp[-q(v-u)^2/2] f_new. Original f_old'=0
            # becomes f_new' - q(v-u) f_new=0 at either cutoff.
            rows[0, 0] -= q * (x[0] - u)
            rows[1, -1] -= q * (x[-1] - u)
        elif boundary != 'gauge_neumann':
            raise ValueError('unknown boundary')
        extension[[0, -1]] = solve(rows[:, [0, points-1]], -rows[:, 1:-1])
        extensions.append(extension)
    blocks = []
    off = []
    for index, sign in enumerate((1., -1.)):
        u = sign * omega
        potential = gamma*q*x*(x-u) + gamma*ta*(q*q*(x-u)**2-q)
        drift = -gamma*x - 2*gamma*ta*q*(x-u)
        continuous = gamma*ta*second + drift[:, None]*derivative + np.diag(potential)
        block = (continuous @ extensions[index])[1:-1]
        jump = np.zeros((size, size))
        for i, v in enumerate(x[1:-1]):
            for lo, hi in ((-width, v), (v, width)):
                gas = (lo+hi)/2 + (hi-lo)*nodes/2
                density = np.exp(-(gas-u)**2/(2*tb))/np.sqrt(2*np.pi*tb)
                rates = 3.6*((1-eta)+eta*abs(v-gas))*density*weights*(hi-lo)/2
                heat = .5*((v-u)**2-(gas-u)**2)
                evaluation = interpolation(x, barycentric, gas) @ extensions[index]
                jump[i] += (rates*np.exp(-(chi-q)*heat)) @ evaluation
                jump[i, i] -= rates.sum()
        blocks.append(block+jump-flip*np.eye(size))
        off.append(flip*np.exp(-2*q*sign*omega*x[1:-1])[:, None]
                   * extensions[1-index][1:-1])
    generator = np.block([[blocks[0], off[0]], [off[1], blocks[1]]])
    values = eigvals(generator, check_finite=False)
    ordered = values[np.argsort(-values.real)]
    principal = ordered[0]
    return dict(value=float(principal.real), imaginary=float(abs(principal.imag)),
                spectral_gap=float(principal.real-ordered[1].real),
                gauge=float(q), points=points, extent=extent,
                quadrature=quadrature, boundary=boundary)
