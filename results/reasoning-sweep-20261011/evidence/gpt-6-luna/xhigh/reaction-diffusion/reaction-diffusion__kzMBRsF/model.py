"""Transport and equilibrium fluctuation model for the electrolyte cell."""

import numpy as np
from scipy.fft import dct, idct
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from scipy.sparse import diags


# A+ + B+ + 2 C- <-> P
ASSOCIATION = 0.5
NEUTRAL_MOBILITY = 0.15
CELL_LENGTH = 0.001


def equilibrium(total):
    """Return free A, free B, free C, and P at local association equilibrium.

    ``total`` contains the total A and B constituent concentrations.  The
    scalar bound concentration is the unique root in [0, min(A, B)] of the
    mass-action equation.  A safeguarded Newton iteration solves it for each
    point independently.
    """
    total = np.asarray(total, dtype=float)
    if total.shape == () or total.shape[-1] != 2:
        raise ValueError("total must have final dimension 2 (A and B)")
    if not np.all(np.isfinite(total)):
        raise ValueError("total concentrations must be finite")

    # Nonnegative concentrations are required by the equilibrium relation.
    a = np.maximum(total[..., 0], 0.0)
    b = np.maximum(total[..., 1], 0.0)
    upper = np.minimum(a, b)
    bound = np.zeros_like(upper)

    for _ in range(48):
        free_a = a - bound
        free_b = b - bound
        free_c = free_a + free_b
        residual = bound - ASSOCIATION * free_a * free_b * free_c**2
        derivative = 1.0 + ASSOCIATION * (
            free_c**3 + 4.0 * free_a * free_b * free_c
        )
        updated = np.clip(bound - residual / derivative, 0.0, upper)
        if not np.any(np.abs(updated - bound) > 2e-14):
            bound = updated
            break
        bound = updated

    free_a = a - bound
    free_b = b - bound
    return np.stack((free_a, free_b, free_a + free_b, bound), axis=-1)


def _check_times(times):
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError("t must be a finite one-dimensional array")
    if np.any(times < 0.0) or np.any(np.diff(times) < 0.0):
        raise ValueError("t must be increasing and nonnegative")
    return times


def _check_mesh(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x)):
        raise ValueError("x must be a finite one-dimensional mesh with at least two points")
    spacing = np.diff(x)
    if np.any(spacing <= 0.0) or not np.allclose(spacing, spacing[0], rtol=1e-8, atol=1e-14):
        raise ValueError("x must be equally spaced and strictly increasing")
    return x


def _binary_prediction(times, x, initial, diffusivity):
    """Exact semidiscrete solution for a preparation with one cation only."""
    intervals = len(x) - 1
    dx = x[1] - x[0]
    mode = np.arange(intervals + 1)
    spatial_rate = 4.0 / dx**2 * np.sin(mode * np.pi / (2.0 * intervals)) ** 2
    # Zero-current Nernst-Planck transport gives effective rates 4D/3 for A
    # and 8D/3 for B when the corresponding anion concentration equals it.
    rate_factor = np.array([4.0 / 3.0, 8.0 / 3.0])
    coefficients = dct(initial, type=1, axis=0)
    rates = diffusivity * spatial_rate[:, None] * rate_factor[None, :]
    evolved = np.exp(-times[:, None, None] * rates[None, :, :])
    return idct(evolved * coefficients[None, :, :], type=1, axis=1)


class TransportModel:
    """Fit and use the common ionic diffusion scale for the cell."""

    def __init__(self):
        self.diffusivity = None

    def fit(self, data):
        """Fit D by weighted least squares to the unary calibration runs."""
        x = _check_mesh(data["x"])
        times = _check_times(data["t"])
        initial = np.asarray(data["initial"], dtype=float)
        observed = np.asarray(data["concentration"], dtype=float)
        sigma = np.asarray(data["sigma"], dtype=float)

        if initial.ndim != 3 or initial.shape[1:] != (len(x), 2):
            raise ValueError("initial must have shape (preparations, len(x), 2)")
        expected = (initial.shape[0], len(times), len(x), 2)
        if observed.shape != expected or sigma.shape != expected:
            raise ValueError("concentration and sigma must have shape (preparations, len(t), len(x), 2)")
        if not (np.all(np.isfinite(initial)) and np.all(np.isfinite(observed))):
            raise ValueError("calibration concentrations must be finite")
        if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise ValueError("sigma must contain finite positive standard deviations")
        if not len(times) or not initial.shape[0]:
            raise ValueError("calibration data must contain times and preparations")

        intervals = len(x) - 1
        dx = x[1] - x[0]
        modes = np.arange(intervals + 1)
        spatial_rate = 4.0 / dx**2 * np.sin(modes * np.pi / (2.0 * intervals)) ** 2

        # Calibration preparations contain just A or just B. Cache their DCT
        # coefficients so each trial value of D only needs a cheap mode decay.
        runs = []
        for run in range(initial.shape[0]):
            channel_mass = np.max(np.abs(initial[run]), axis=0)
            channel = int(np.argmax(channel_mass))
            if channel_mass[channel] == 0.0:
                continue
            other = 1 - channel
            if np.any(np.abs(initial[run, :, other]) > 1e-12):
                raise ValueError("calibration preparations must contain only one cation constituent")
            if np.any(np.abs(initial[run, :, channel]) > 0.0):
                coeff = dct(initial[run, :, channel], type=1)
                active_factor = (4.0 / 3.0) if channel == 0 else (8.0 / 3.0)
                runs.append((coeff, active_factor, observed[run, :, :, channel], sigma[run, :, :, channel]))

        if not runs:
            raise ValueError("calibration data contain no nonzero initial constituent")

        def objective(log_diffusivity):
            diffusivity = np.exp(log_diffusivity)
            squared_error = 0.0
            for coeff, factor, target, uncertainty in runs:
                decay = np.exp(-times[:, None] * (diffusivity * factor * spatial_rate)[None, :])
                prediction = idct(decay * coeff[None, :], type=1, axis=1)
                residual = (prediction - target) / uncertainty
                squared_error += np.sum(residual**2)
            return float(squared_error)

        lower, upper = np.log(1e-16), np.log(1e-5)
        result = minimize_scalar(
            objective,
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 1e-11},
        )
        candidates = (lower, float(result.x), upper)
        best_log_d = min(candidates, key=objective)
        self.diffusivity = float(np.exp(best_log_d))
        return self

    def _binary(self, t, x, initial):
        return _binary_prediction(t, x, initial, self.diffusivity)

    def predict(self, t, x, initial):
        """Predict total A and B profiles on the supplied uniform mesh."""
        if self.diffusivity is None or not np.isfinite(self.diffusivity) or self.diffusivity <= 0.0:
            raise RuntimeError("fit the model before calling predict")
        times = _check_times(t)
        x = _check_mesh(x)
        initial = np.asarray(initial, dtype=float)
        if initial.shape != (len(x), 2) or not np.all(np.isfinite(initial)):
            raise ValueError("initial must be a finite array with shape (len(x), 2)")
        if np.any(initial < 0.0):
            raise ValueError("initial constituent concentrations must be nonnegative")
        if not np.isclose(x[0], 0.0) or not np.isclose(x[-1], CELL_LENGTH):
            raise ValueError("x must span the cell from 0 to 0.001 m")
        if not len(times):
            return np.empty((0, len(x), 2), dtype=float)
        if times[-1] == 0.0:
            return initial[None, :, :].copy()

        if np.all(initial[:, 0] == 0.0) or np.all(initial[:, 1] == 0.0):
            return self._binary(times, x, initial)

        n = len(x)
        dx = x[1] - x[0]
        cell_width = np.full(n, dx)
        cell_width[[0, -1]] *= 0.5
        da, db, dc = self.diffusivity * np.array([1.0, 4.0, 2.0])

        def derivative(_time, flat_state):
            # Solver trial stages can stray a tiny amount below zero; clipping
            # only the equilibrium lookup keeps mass action well defined.
            totals = np.maximum(flat_state.reshape(n, 2), 0.0)
            species = equilibrium(totals)
            face = 0.5 * (species[:-1] + species[1:])
            gradient = np.diff(species, axis=0) / dx

            denominator = da * face[:, 0] + db * face[:, 1] + dc * face[:, 2]
            denominator = np.maximum(denominator, np.finfo(float).tiny)
            field_gradient = -(
                da * gradient[:, 0] + db * gradient[:, 1] - dc * gradient[:, 2]
            ) / denominator

            flux_a = -da * (gradient[:, 0] + face[:, 0] * field_gradient)
            flux_b = -db * (gradient[:, 1] + face[:, 1] * field_gradient)
            flux_p = -self.diffusivity * NEUTRAL_MOBILITY * gradient[:, 3]

            total_flux = np.zeros((n + 1, 2), dtype=float)
            total_flux[1:-1, 0] = flux_a + flux_p
            total_flux[1:-1, 1] = flux_b + flux_p
            return (-np.diff(total_flux, axis=0) / cell_width[:, None]).ravel()

        # Each node couples to itself and its immediate spatial neighbors in
        # both constituent channels.
        nstate = 2 * n
        offsets = range(-3, 4)
        sparsity = diags(
            [np.ones(nstate - abs(offset)) for offset in offsets],
            list(offsets),
            shape=(nstate, nstate),
            format="csr",
        )
        result = solve_ivp(
            derivative,
            (0.0, float(times[-1])),
            initial.ravel(),
            method="BDF",
            t_eval=times,
            rtol=1e-8,
            atol=1e-10,
            jac_sparsity=sparsity,
        )
        if not result.success:
            raise RuntimeError(result.message)
        return result.y.T.reshape(len(times), n, 2)

    def correlate(self, t, totals, harmonic=1):
        """Return the full two-time covariance matrix for one cosine mode."""
        if self.diffusivity is None or not np.isfinite(self.diffusivity) or self.diffusivity <= 0.0:
            raise RuntimeError("fit the model before calling correlate")
        times = _check_times(t)
        totals = np.asarray(totals, dtype=float)
        if totals.shape != (2,) or not np.all(np.isfinite(totals)):
            raise ValueError("totals must be a finite pair (A, B)")
        if np.any(totals < 0.2) or np.any(totals > 2.0):
            raise ValueError("each mean total concentration must be in [0.2, 2] mol m^-3")
        if not isinstance(harmonic, (int, np.integer)) or not 1 <= harmonic <= 8:
            raise ValueError("harmonic must be an integer from 1 through 8")
        return equilibrium_correlations(times, totals, harmonic, self.diffusivity)


def fluctuation_matrices(totals, diffusivity):
    """Build the electroneutral susceptibility and constituent mobility."""
    concentration = equilibrium(np.asarray(totals, dtype=float)[None, :])[0]
    charge = np.array([1.0, 1.0, -1.0, 0.0])
    constituents = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]])

    # Ideal species-number fluctuations conditioned on zero charge fluctuation.
    population = np.diag(concentration)
    charge_variance = charge @ population @ charge
    conditional = population - np.outer(population @ charge, population @ charge) / charge_variance
    susceptibility = constituents @ conditional @ constituents.T

    # Project the independent Brownian flux mobility onto zero electric current.
    species_diffusivities = diffusivity * np.array([1.0, 4.0, 2.0, NEUTRAL_MOBILITY])
    mobile = np.diag(species_diffusivities * concentration)
    mobile_charge = mobile @ charge
    charge_mobility = charge @ mobile @ charge
    constrained = mobile - np.outer(mobile_charge, mobile_charge) / charge_mobility
    mobility = constituents @ constrained @ constituents.T
    return susceptibility, mobility


def equilibrium_correlations(t, totals, harmonic, diffusivity):
    """Evaluate the resolved equilibrium-mode two-time covariance."""
    susceptibility, mobility = fluctuation_matrices(totals, diffusivity)
    wave_number = harmonic * np.pi / CELL_LENGTH
    relaxation = wave_number**2 * mobility @ np.linalg.inv(susceptibility)

    # The measurement convention specifies independent ideal total-constituent
    # fluctuations at equal time, in the documented A,B channel order.
    equal_time_covariance = np.diag(totals)
    return np.array(
        [expm(-float(time) * relaxation) @ equal_time_covariance for time in t]
    ).reshape(len(t), 2, 2)
