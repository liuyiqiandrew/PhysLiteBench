"""Electroneutral transport and equilibrium fluctuations in the cell."""

import numpy as np
from scipy.fft import dct, idct
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from scipy.sparse import diags


LENGTH = 0.001
ASSOCIATION = 0.5
RELATIVE_DIFFUSIVITIES = np.array([1.0, 4.0, 2.0, 0.15])
CHARGE = np.array([1.0, 1.0, -1.0, 0.0])
# Rows give the number of A and B constituents in each species (A, B, C, P).
CONSTITUENTS = np.array([[1.0, 0.0, 0.0, 1.0],
                         [0.0, 1.0, 0.0, 1.0]])


def equilibrium(totals):
    """Return local (free A, free B, free C, P) concentrations."""
    totals = np.asarray(totals, dtype=float)
    a, b = totals[..., 0], totals[..., 1]
    bound = np.zeros_like(a)
    for _ in range(50):
        free_a, free_b = a - bound, b - bound
        free_c = free_a + free_b
        residual = bound - ASSOCIATION * free_a * free_b * free_c**2
        slope = 1 + ASSOCIATION * (free_c**3 + 4 * free_a * free_b * free_c)
        updated = np.clip(bound - residual / slope, 0.0, np.minimum(a, b))
        if np.all(np.abs(updated - bound) < 1e-14):
            bound = updated
            break
        bound = updated
    return np.stack((a - bound, b - bound, a + b - 2 * bound, bound), axis=-1)


def _spatial_rates(x):
    """Eigenvalues of the nodal, no-flux finite-volume Laplacian."""
    intervals = len(x) - 1
    return 4 * np.sin(np.arange(intervals + 1) * np.pi / (2 * intervals))**2 / (x[1] - x[0])**2


class TransportModel:
    def __init__(self):
        self.diffusivity = None

    def fit(self, data):
        """Fit the one diffusion scale by Gaussian weighted least squares."""
        x = np.asarray(data["x"], dtype=float)
        t = np.asarray(data["t"], dtype=float)
        initial = np.asarray(data["initial"], dtype=float)
        measured = np.asarray(data["concentration"], dtype=float)
        sigma = np.asarray(data["sigma"], dtype=float)

        # Every calibration preparation contains a single cation constituent.
        # Its counterion gives ambipolar diffusivity 4D/3 (A) or 8D/3 (B).
        coefficients = dct(initial, type=1, axis=1)
        exponent = (t[None, :, None, None]
                    * _spatial_rates(x)[None, None, :, None]
                    * np.array([4 / 3, 8 / 3])[None, None, None, :])

        def objective(log_diffusivity):
            profile = idct(coefficients[:, None, :, :]
                           * np.exp(-10.0**log_diffusivity * exponent),
                           type=1, axis=2)
            return np.sum(((profile - measured) / sigma)**2)

        # A coarse scan avoids choosing a flat tail of the objective when the
        # diffusion time is far outside the observed time window.
        grid = np.linspace(-16.0, -3.0, 53)
        scores = np.array([objective(value) for value in grid])
        best = int(np.argmin(scores))
        left, right = grid[max(0, best - 1)], grid[min(len(grid) - 1, best + 1)]
        optimum = minimize_scalar(objective, bounds=(left, right), method="bounded",
                                  options={"xatol": 1e-11})
        self.diffusivity = float(10.0**optimum.x)
        return self

    def _binary(self, t, x, initial):
        coefficients = dct(initial, type=1, axis=0)
        rates = (self.diffusivity * _spatial_rates(x)[:, None]
                 * np.array([4 / 3, 8 / 3])[None, :])
        return idct(coefficients[None, :, :] * np.exp(-t[:, None, None] * rates[None, :, :]),
                    type=1, axis=1)

    def predict(self, t, x, initial):
        """Predict total A and B concentrations at the requested mesh and times."""
        t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float)
        initial = np.asarray(initial, dtype=float)
        if t[-1] == 0:
            return initial[None, :, :].copy()
        if np.all(initial[:, 0] == 0) or np.all(initial[:, 1] == 0):
            return self._binary(t, x, initial)
        if np.all(initial == initial[0]):
            return np.broadcast_to(initial, (len(t), *initial.shape)).copy()

        nodes = len(x)
        dx = x[1] - x[0]
        widths = np.full(nodes, dx)
        widths[[0, -1]] *= 0.5
        diffusion = self.diffusivity * RELATIVE_DIFFUSIVITIES

        def derivative(_time, flattened):
            species = equilibrium(flattened.reshape(nodes, 2))
            face = (species[:-1] + species[1:]) / 2
            gradient = np.diff(species, axis=0) / dx
            # Zero charge current determines the electric potential gradient.
            charge_gradient = gradient[:, :3] @ (diffusion[:3] * CHARGE[:3])
            conductivity = face[:, :3] @ diffusion[:3]
            electric_gradient = np.zeros_like(conductivity)
            np.divide(-charge_gradient, conductivity, out=electric_gradient,
                      where=conductivity > 0)
            ionic_flux = -diffusion[:2] * (
                gradient[:, :2] + face[:, :2] * electric_gradient[:, None])
            cluster_flux = -diffusion[3] * gradient[:, 3]
            flux = np.zeros((nodes + 1, 2))
            flux[1:-1] = ionic_flux + cluster_flux[:, None]
            return (-np.diff(flux, axis=0) / widths[:, None]).ravel()

        jacobian_pattern = diags(
            [np.ones(2 * nodes - abs(offset)) for offset in range(-3, 4)],
            range(-3, 4), format="csr")
        solution = solve_ivp(derivative, (0, t[-1]), initial.ravel(),
                             method="BDF", t_eval=t, rtol=1e-8, atol=1e-10,
                             jac_sparsity=jacobian_pattern)
        if not solution.success:
            raise RuntimeError(solution.message)
        return solution.y.T.reshape(len(t), nodes, 2)

    def correlate(self, t, totals, harmonic=1):
        """Two-time covariance of a nonzero cosine mode, in A,B order."""
        t = np.asarray(t, dtype=float)
        susceptibility, mobility = fluctuation_matrices(totals, self.diffusivity)
        # Generalized eigenvectors satisfy V.T @ susceptibility @ V = I.
        # This form keeps S(t) symmetric and gives S(0)=susceptibility.
        rates, vectors = eigh(mobility, susceptibility)
        amplitudes = susceptibility @ vectors
        decays = np.exp(-t[:, None] * (harmonic * np.pi / LENGTH)**2 * rates[None, :])
        return np.einsum("ik,tk,jk->tij", amplitudes, decays, amplitudes)


def fluctuation_matrices(totals, diffusivity):
    """Static constituent covariance and charge-constrained flux mobility."""
    species = equilibrium(np.asarray(totals, dtype=float))
    population = np.diag(species)
    # Independent ideal species populations, conditioned on local neutrality.
    charge_population = population @ CHARGE
    conditional_population = (population - np.outer(charge_population, charge_population)
                              / (CHARGE @ charge_population))
    susceptibility = CONSTITUENTS @ conditional_population @ CONSTITUENTS.T

    # The electric field also cancels the Brownian part of the charge flux.
    species_mobility = np.diag(diffusivity * RELATIVE_DIFFUSIVITIES * species)
    charge_mobility = species_mobility @ CHARGE
    constrained_mobility = (species_mobility
                            - np.outer(charge_mobility, charge_mobility)
                            / (CHARGE @ charge_mobility))
    mobility = CONSTITUENTS @ constrained_mobility @ CONSTITUENTS.T
    return susceptibility, mobility
