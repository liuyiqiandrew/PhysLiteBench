from functools import lru_cache
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


@lru_cache(128)
def fields(amplitudes, phase, points=63):
    q = np.arange(points)*2*np.pi/points
    x, y = np.meshgrid(q, q, indexing='ij')
    a, b, c, d = amplitudes
    first = 1+a*np.cos(x)+b*np.cos(y)
    second = 1+c*np.cos(x+phase)+d*np.cos(x+y)
    concentrations = np.array([first, second, first+second])
    ratios = np.array([1., 4., 1.])
    charges = np.array([1., 1., -1.])
    conductivity = np.einsum('i,ijk->jk', ratios, concentrations)
    diffusion_charge = np.einsum('i,ijk->jk', ratios*charges, concentrations)
    modes = np.fft.fftfreq(points, 1/points)
    kx, ky = np.meshgrid(modes, modes, indexing='ij')
    k2 = kx*kx+ky*ky
    def gradient(value):
        transformed = np.fft.fft2(value)
        return np.array([np.fft.ifft2(1j*kx*transformed).real,
                         np.fft.ifft2(1j*ky*transformed).real])
    def divergence(value):
        return np.fft.ifft2(1j*kx*np.fft.fft2(value[0])+1j*ky*np.fft.fft2(value[1])).real
    def action(value):
        grid = value.reshape(points, points)
        return (-divergence(conductivity*gradient(grid))+grid.mean()).ravel()
    denominator = conductivity.mean()*k2
    denominator[0,0] = 1.
    def precondition(value):
        grid = value.reshape(points, points)
        return np.fft.ifft2(np.fft.fft2(grid)/denominator).real.ravel()
    operator = LinearOperator((points*points, points*points), matvec=action, dtype=float)
    preconditioner = LinearOperator(operator.shape, matvec=precondition, dtype=float)
    def solve(rhs):
        result, status = cg(operator, rhs.ravel(), M=preconditioner,
                            rtol=3e-13, atol=1e-14, maxiter=1000)
        if status != 0:
            raise RuntimeError('Potential solve did not converge.')
        return result.reshape(points, points)
    laplacian = np.fft.ifft2(-k2*np.fft.fft2(diffusion_charge)).real
    electric = -gradient(solve(laplacian))
    gradients = np.array([gradient(value) for value in concentrations])
    flux = -ratios[:,None,None,None]*gradients
    flux += (ratios*charges)[:,None,None,None]*concentrations[:,None]*electric
    rates = np.array([-divergence(current) for current in flux])
    return x, y, concentrations, electric, flux, rates


@lru_cache(512)
def unit_signal(amplitudes, phase, species, detector):
    x, y, _, _, _, rates = fields(amplitudes, phase)
    m, n = detector
    weight = np.cos(m*x+n*y+np.pi/4)
    return float(2*np.mean(rates[species]*weight))


def predict_at(experiments, diffusivity):
    return np.array([unit_signal(tuple(e['amplitudes']), e['phase'], e['species'], tuple(e['detector']))
                     for e in experiments])*diffusivity


class Model:
    def __init__(self):
        self.diffusivity = 1.

    def fit(self, records):
        records = list(records)
        basis = predict_at([r['input'] for r in records], 1.)
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        self.diffusivity = float(np.clip(np.sum(basis*values/sigma**2)/
                                       np.sum(basis*basis/sigma**2), .5, 2.))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
