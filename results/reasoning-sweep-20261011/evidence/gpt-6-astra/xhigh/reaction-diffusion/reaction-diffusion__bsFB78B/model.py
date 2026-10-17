"""Electroneutral transport and equilibrium fluctuations of an associating salt."""

import numpy as np
from scipy.fft import dct, idct
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from scipy.sparse import diags

ASSOCIATION = 0.5
NEUTRAL_MOBILITY = 0.15
CELL_LENGTH = 0.001
RELATIVE_DIFFUSIVITIES = np.array([1.0, 4.0, 2.0, NEUTRAL_MOBILITY])
CHARGE = np.array([1.0, 1.0, -1.0, 0.0])
CONSTITUENTS = np.array([[1.0, 0.0, 0.0, 1.0],
                         [0.0, 1.0, 0.0, 1.0]])


def equilibrium(total):
    """Return free A, free B, free C, and P, with species on the last axis.

    The mass-action equation has exactly one root on [0, min(total)].
    Newton iteration from zero approaches that root from below.
    """
    total = np.asarray(total, dtype=float)
    a, b = total[..., 0], total[..., 1]
    bound = np.zeros_like(a)
    for _ in range(64):
        free_a, free_b = a - bound, b - bound
        anion = free_a + free_b
        residual = bound - ASSOCIATION * free_a * free_b * anion**2
        slope = 1.0 + ASSOCIATION * (anion**3 + 4 * free_a * free_b * anion)
        updated = np.clip(bound - residual / slope, 0.0, np.minimum(a, b))
        converged = np.all(np.abs(updated - bound) <=
                           4 * np.finfo(float).eps * np.maximum(1.0, np.minimum(a, b)))
        bound = updated
        if converged:
            break
    return np.stack((a - bound, b - bound, a + b - 2 * bound, bound), axis=-1)


def fluctuation_matrices(totals, diffusivity):
    """Return the constituent susceptibility and Onsager mobility.

    Ideal species have Poisson concentration covariance diag(c). Conditioning
    on zero local charge gives its Schur complement. Mapping this covariance
    to conserved constituents includes the shared cluster fluctuations;
    reaction noise itself has zero projection onto these constituents.

    Zero instantaneous electric current projects the species Brownian fluxes
    in the same way, but with D_i*c_i in place of c_i. Thus the resulting
    constituent noise covariance is twice the mobility below.
    """
    concentration = equilibrium(totals)
    population_charge = concentration * CHARGE
    conditional = (np.diag(concentration) -
                   np.outer(population_charge, population_charge) /
                   (CHARGE @ population_charge))
    susceptibility = CONSTITUENTS @ conditional @ CONSTITUENTS.T

    mobile = diffusivity * RELATIVE_DIFFUSIVITIES * concentration
    mobile_charge = mobile * CHARGE
    constrained = (np.diag(mobile) - np.outer(mobile_charge, mobile_charge) /
                   (CHARGE @ mobile_charge))
    mobility = CONSTITUENTS @ constrained @ CONSTITUENTS.T
    return susceptibility, mobility


def equilibrium_correlations(t, totals, harmonic, diffusivity):
    """Evaluate exp(-q^2 M S^-1 t) S in symmetric, whitened coordinates."""
    covariance, mobility = fluctuation_matrices(totals, diffusivity)
    root = np.linalg.cholesky(covariance)
    # G = root^-1 M root^-T is symmetric positive definite. Using its
    # eigensystem avoids a nonsymmetric exponential and preserves reciprocity.
    left = np.linalg.solve(root, mobility)
    generator = np.linalg.solve(root, left.T).T
    rates, vectors = np.linalg.eigh((generator + generator.T) / 2)
    modes = root @ vectors
    wave_number = harmonic * np.pi / CELL_LENGTH
    decay = np.exp(-np.asarray(t)[:, None] * wave_number**2 * rates[None, :])
    return np.einsum('ik,tk,jk->tij', modes, decay, modes)


def _times(t):
    t = np.asarray(t, dtype=float)
    if (t.ndim != 1 or not np.isfinite(t).all() or np.any(t < 0) or
            np.any(np.diff(t) <= 0)):
        raise ValueError('Times must be a strictly increasing nonnegative array.')
    return t


def _spatial_rates(x):
    intervals = len(x) - 1
    return 4 / (x[1] - x[0])**2 * np.sin(
        np.arange(intervals + 1) * np.pi / (2 * intervals))**2


class TransportModel:
    def __init__(self):
        self.diffusivity = None

    def _require_fitted(self):
        if self.diffusivity is None or not np.isfinite(self.diffusivity) or self.diffusivity <= 0:
            raise ValueError('Fit the model before making predictions.')

    def fit(self, data):
        """Fit the common diffusion scale by Gaussian maximum likelihood.

        Each calibration contains only A/C or B/C. Eliminating the electric
        field gives ambipolar diffusivities 4D/3 and 8D/3, respectively.
        Reflecting diffusion on the mesh is diagonal in the DCT-I basis.
        """
        x = np.asarray(data['x'], dtype=float)
        t = _times(data['t'])
        initial = np.asarray(data['initial'], dtype=float)
        observed = np.asarray(data['concentration'], dtype=float)
        sigma = np.asarray(data['sigma'], dtype=float)
        if np.any(np.any(initial[..., 0] != 0, axis=1) &
                  np.any(initial[..., 1] != 0, axis=1)):
            raise ValueError('Calibration preparations must contain a single cation constituent.')
        if np.any(sigma <= 0) or not np.isfinite(sigma).all():
            raise ValueError('Measurement standard deviations must be finite and positive.')
        if not len(t) or t[-1] <= 0:
            raise ValueError('Calibration must include positive observation times.')

        coefficients = dct(initial, type=1, axis=1)
        rates = _spatial_rates(x)[:, None] * np.array([4 / 3, 8 / 3])[None, :]
        scaled_times = t[:, None, None] * rates[None, :, :]
        scale = (x[-1] - x[0])**2 / t[-1]

        def objective(log_scale):
            diffusivity = scale * np.exp(log_scale)
            predicted = idct(coefficients[:, None, :, :] *
                             np.exp(-diffusivity * scaled_times)[None, :, :, :],
                             type=1, axis=2)
            residual = (predicted - observed) / sigma
            return np.sum(residual**2)

        fitted = minimize_scalar(objective, bounds=(-20.0, 20.0), method='bounded',
                                 options={'xatol': 1e-11})
        if not fitted.success or not np.isfinite(fitted.fun):
            raise RuntimeError('Diffusivity calibration failed.')
        self.diffusivity = float(scale * np.exp(fitted.x))
        return self

    def _binary(self, t, x, initial):
        coefficients = dct(initial, type=1, axis=0)
        rates = (_spatial_rates(x)[:, None] * self.diffusivity *
                 np.array([4 / 3, 8 / 3])[None, :])
        return idct(np.exp(-t[:, None, None] * rates[None, :, :]) *
                    coefficients[None, :, :], type=1, axis=1)

    def predict(self, t, x, initial):
        """Evolve total constituents with local chemical equilibrium enforced."""
        self._require_fitted()
        t = _times(t)
        x, initial = np.asarray(x, dtype=float), np.asarray(initial, dtype=float)
        if x.ndim != 1 or len(x) < 2 or initial.shape != (len(x), 2):
            raise ValueError('Expected a spatial mesh and initial shape (len(x), 2).')
        dx = x[1] - x[0]
        if dx <= 0 or not np.allclose(np.diff(x), dx, rtol=1e-8, atol=1e-14):
            raise ValueError('The spatial mesh must be increasing and equally spaced.')
        if not np.isfinite(initial).all() or np.any(initial < 0):
            raise ValueError('Initial total concentrations must be finite and nonnegative.')
        if not len(t):
            return np.empty((0, len(x), 2))
        if t[-1] == 0:
            return initial[None, :, :].copy()
        if np.all(initial[:, 0] == 0) or np.all(initial[:, 1] == 0):
            return self._binary(t, x, initial)

        count = len(x)
        # Endpoint control volumes have half width: their weights also give
        # the exact integrals of the piecewise-linear initial preparations.
        width = np.full(count, dx)
        width[[0, -1]] *= 0.5
        ion_diffusivities = self.diffusivity * RELATIVE_DIFFUSIVITIES[:3]

        def derivative(time, flat):
            concentration = equilibrium(flat.reshape(count, 2))
            face = (concentration[:-1] + concentration[1:]) / 2
            gradient = np.diff(concentration, axis=0) / dx
            conductivity = face[:, :3] @ ion_diffusivities
            charge_gradient = gradient[:, :3] @ (ion_diffusivities * CHARGE[:3])
            potential_gradient = np.divide(-charge_gradient, conductivity,
                                           out=np.zeros_like(conductivity),
                                           where=conductivity > 0)
            free_flux = -ion_diffusivities[:2] * (
                gradient[:, :2] + face[:, :2] * potential_gradient[:, None])
            bound_flux = -self.diffusivity * NEUTRAL_MOBILITY * gradient[:, 3]
            flux = np.zeros((count + 1, 2))
            flux[1:-1] = free_flux + bound_flux[:, None]
            return (-np.diff(flux, axis=0) / width[:, None]).ravel()

        sparsity = diags([np.ones(2 * count - abs(i)) for i in range(-3, 4)],
                         range(-3, 4), format='csr')
        result = solve_ivp(derivative, (0, t[-1]), initial.ravel(), method='BDF',
                           t_eval=t, rtol=1e-8, atol=1e-10, jac_sparsity=sparsity)
        if not result.success:
            raise RuntimeError(result.message)
        return result.y.T.reshape(len(t), count, 2)

    def correlate(self, t, totals, harmonic=1):
        """Return the stationary two-time covariance of a nonzero cosine mode."""
        self._require_fitted()
        t = _times(t)
        totals = np.asarray(totals, dtype=float)
        if totals.shape != (2,) or not np.isfinite(totals).all() or np.any(totals <= 0):
            raise ValueError('Provide two finite, positive total concentrations.')
        if not np.isscalar(harmonic) or harmonic not in range(1, 9):
            raise ValueError('The harmonic must be an integer from 1 through 8.')
        return equilibrium_correlations(t, totals, int(harmonic), self.diffusivity)
