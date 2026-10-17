"""Transport model for the electroneutral associating electrolyte cell.

The state variables used by this module are the *total* A and B constituent
concentrations. The fast reaction is eliminated by solving its local
mass-action equilibrium at every spatial point.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct, idct
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from scipy.sparse import diags


# p = K a b c^2, and D_P / D_A.
ASSOCIATION = 0.5
NEUTRAL_MOBILITY = 0.15
CELL_LENGTH = 1.0e-3


def equilibrium(total: np.ndarray) -> np.ndarray:
    """Return free A, free B, free C, and P at total A/B concentrations.

    ``total`` has a final dimension of length two. The physical root is in
    ``[0, min(a_total, b_total)]``. A safeguarded Newton iteration is used so
    that the result remains reliable for both pure and strongly mixed states.
    """
    totals = np.asarray(total, dtype=float)
    if totals.shape[-1] != 2:
        raise ValueError("total concentrations must have a final dimension of length 2")
    if not np.isfinite(totals).all() or np.any(totals < 0.0):
        raise ValueError("total concentrations must be finite and non-negative")

    a = totals[..., 0]
    b = totals[..., 1]
    upper = np.minimum(a, b)
    bound = np.zeros_like(a)
    lo = np.zeros_like(a)
    hi = upper.copy()

    for _ in range(48):
        free_a = a - bound
        free_b = b - bound
        anion = free_a + free_b
        value = bound - ASSOCIATION * free_a * free_b * anion**2
        derivative = 1.0 + ASSOCIATION * (anion**3 + 4.0 * free_a * free_b * anion)
        candidate = bound - value / derivative
        # f is increasing. Maintain a bracket as insurance against a Newton
        # overshoot near the end of the physical interval.
        lo = np.where(value < 0.0, bound, lo)
        hi = np.where(value > 0.0, bound, hi)
        candidate = np.where((candidate > lo) & (candidate < hi), candidate, 0.5 * (lo + hi))
        if np.max(np.abs(candidate - bound), initial=0.0) < 2.0e-14:
            bound = candidate
            break
        bound = candidate

    bound = np.clip(bound, 0.0, upper)
    return np.stack((a - bound, b - bound, a + b - 2.0 * bound, bound), axis=-1)


def _equilibrium_jacobian(totals: np.ndarray, concentrations: np.ndarray | None = None) -> np.ndarray:
    """Derivative of [free A, free B, free C, P] with respect to totals."""
    u = np.asarray(totals, dtype=float)
    c = equilibrium(u) if concentrations is None else concentrations
    af, bf, cf, _ = np.moveaxis(c, -1, 0)
    denom = 1.0 + ASSOCIATION * (cf**3 + 4.0 * af * bf * cf)
    # Implicit differentiation of p-K(a-p)(b-p)(a+b-2p)^2 = 0.
    pa = ASSOCIATION * (bf * cf**2 + 2.0 * af * bf * cf) / denom
    pb = ASSOCIATION * (af * cf**2 + 2.0 * af * bf * cf) / denom

    result = np.empty(u.shape[:-1] + (4, 2), dtype=float)
    result[..., 0, 0] = 1.0 - pa
    result[..., 0, 1] = -pb
    result[..., 1, 0] = -pa
    result[..., 1, 1] = 1.0 - pb
    result[..., 2, 0] = 1.0 - 2.0 * pa
    result[..., 2, 1] = 1.0 - 2.0 * pb
    result[..., 3, 0] = pa
    result[..., 3, 1] = pb
    return result


def _check_mesh_and_times(t: np.ndarray, x: np.ndarray, initial: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    initial = np.asarray(initial, dtype=float)
    if t.ndim != 1 or x.ndim != 1 or len(x) < 2:
        raise ValueError("t and x must be one-dimensional, with at least two x points")
    if initial.shape != (len(x), 2):
        raise ValueError("initial must have shape (len(x), 2)")
    if not np.isfinite(t).all() or not np.isfinite(x).all() or not np.isfinite(initial).all():
        raise ValueError("t, x, and initial must be finite")
    if np.any(t < 0.0) or np.any(np.diff(t) < 0.0):
        raise ValueError("t must be increasing and non-negative")
    dx = np.diff(x)
    if np.any(dx <= 0.0) or not np.allclose(dx, dx[0], rtol=2e-11, atol=1e-15):
        raise ValueError("x must be an equally spaced increasing mesh")
    if np.any(initial < 0.0):
        raise ValueError("initial total concentrations must be non-negative")
    return t, x, initial


class TransportModel:
    def __init__(self):
        self.diffusivity: float | None = None

    @staticmethod
    def _binary_prediction(t: np.ndarray, x: np.ndarray, initial: np.ndarray, diffusivity: float) -> np.ndarray:
        """Exact semi-discrete Neumann solution for a one-cation preparation."""
        intervals = len(x) - 1
        dx = x[1] - x[0]
        eigenvalue = 4.0 / dx**2 * np.sin(np.arange(intervals + 1) * np.pi / (2.0 * intervals)) ** 2
        modes = dct(initial, type=1, axis=0)
        # Ambipolar diffusivities for (A+,C-) and (B+,C-), respectively.
        rates = eigenvalue[:, None] * diffusivity * np.array([4.0 / 3.0, 8.0 / 3.0])
        return idct(
            np.exp(-t[:, None, None] * rates[None, :, :]) * modes[None, :, :],
            type=1,
            axis=1,
        )

    @staticmethod
    def _is_binary(initial: np.ndarray) -> bool:
        # Exact zero is the documented calibration preparation. The tolerance
        # also avoids invoking the nonlinear solver for harmless roundoff.
        scale = max(1.0, float(np.max(initial, initial=0.0)))
        return bool(np.max(np.abs(initial[:, 0]), initial=0.0) <= 1e-13 * scale or
                    np.max(np.abs(initial[:, 1]), initial=0.0) <= 1e-13 * scale)

    def fit(self, data):
        """Fit the common ionic diffusion scale by weighted least squares."""
        try:
            t = np.asarray(data["t"], dtype=float)
            x = np.asarray(data["x"], dtype=float)
            initial = np.asarray(data["initial"], dtype=float)
            observed = np.asarray(data["concentration"], dtype=float)
            sigma = np.asarray(data["sigma"], dtype=float)
        except (KeyError, TypeError) as exc:
            raise ValueError("data must contain x, t, initial, concentration, and sigma") from exc

        if initial.ndim != 3 or initial.shape[2] != 2 or initial.shape[0] == 0:
            raise ValueError("initial must have shape (records, x, 2), with at least one record")
        for record in initial:
            _check_mesh_and_times(t, x, record)
        expected = (initial.shape[0], len(t), len(x), 2)
        if observed.shape != expected or sigma.shape != expected:
            raise ValueError("concentration and sigma must have shape (records, t, x, 2)")
        if not np.isfinite(observed).all() or not np.isfinite(sigma).all() or np.any(sigma <= 0.0):
            raise ValueError("concentration must be finite and sigma must be finite and positive")
        use_binary = all(self._is_binary(record) for record in initial)

        positive_times = t[t > 0.0]
        reference = (x[-1] - x[0]) ** 2 / (float(np.max(positive_times)) if len(positive_times) else 1.0)
        low = max(reference * 1e-10, np.finfo(float).tiny)
        high = max(reference * 1e10, low * 1e4)

        old_diffusivity = self.diffusivity

        def objective(log_diffusivity: float) -> float:
            value = float(np.exp(log_diffusivity))
            try:
                if use_binary:
                    prediction = np.stack(
                        [self._binary_prediction(t, x, record, value) for record in initial], axis=0
                    )
                else:
                    self.diffusivity = value
                    prediction = np.stack([self.predict(t, x, record) for record in initial], axis=0)
                residual = (prediction - observed) / sigma
                score = float(np.sum(residual * residual))
                return score if np.isfinite(score) else np.inf
            except (FloatingPointError, RuntimeError, ValueError):
                return np.inf

        # At very large D every sampled profile is already uniform. That
        # creates a numerically flat (and therefore not globally unimodal)
        # tail in the likelihood. Locate the physical basin first, then make
        # the accurate bounded scalar minimization inside that basin.
        scan = np.linspace(np.log(low), np.log(high), 61 if use_binary else 13)
        scan_scores = np.array([objective(point) for point in scan])
        best = int(np.argmin(scan_scores))
        left = scan[max(0, best - 1)]
        right = scan[min(len(scan) - 1, best + 1)]
        if left == right:  # Only possible for a one-point scan, kept defensive.
            left, right = np.log(low), np.log(high)
        try:
            result = minimize_scalar(
                objective,
                bounds=(float(left), float(right)),
                method="bounded",
                options={"xatol": 1e-10, "maxiter": 160},
            )
        finally:
            self.diffusivity = old_diffusivity
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError("could not fit a finite positive diffusivity")
        self.diffusivity = float(np.exp(result.x))
        return self

    def _require_diffusivity(self) -> float:
        if self.diffusivity is None or not np.isfinite(self.diffusivity) or self.diffusivity <= 0.0:
            raise RuntimeError("fit must be called before prediction")
        return float(self.diffusivity)

    def predict(self, t, x, initial):
        """Predict total A/B profiles for a prepared cell."""
        t, x, initial = _check_mesh_and_times(t, x, initial)
        diffusivity = self._require_diffusivity()
        if len(t) == 0:
            return np.empty((0, len(x), 2), dtype=float)
        if self._is_binary(initial):
            return self._binary_prediction(t, x, initial, diffusivity)
        if t[-1] == 0.0:
            return np.broadcast_to(initial, (len(t),) + initial.shape).copy()

        n = len(x)
        dx = x[1] - x[0]
        # Endpoint control volumes are half-widths, so integrated constituent
        # amounts are conserved by the no-flux finite-volume update.
        widths = np.full(n, dx)
        widths[[0, -1]] *= 0.5
        species_diffusivity = diffusivity * np.array([1.0, 4.0, 2.0, NEUTRAL_MOBILITY])
        charge_weighted_diffusivity = np.array([species_diffusivity[0], species_diffusivity[1], -species_diffusivity[2]])

        def rhs(_time: float, flat: np.ndarray) -> np.ndarray:
            totals = flat.reshape(n, 2)
            # Implicit trial iterates can make a concentration infinitesimally
            # negative at a sharp zero-concentration front. Extend the flux
            # smoothly enough there by evaluating it at the physical boundary;
            # accepted solutions remain non-negative to solver tolerance.
            physical_totals = np.maximum(totals, 0.0)
            face_totals = 0.5 * (physical_totals[:-1] + physical_totals[1:])
            face_concentrations = equilibrium(face_totals)
            jacobian = _equilibrium_jacobian(face_totals, face_concentrations)
            concentration_gradient = np.einsum(
                "...ij,...j->...i", jacobian, np.diff(physical_totals, axis=0) / dx
            )

            denom = np.einsum("i,...i->...", species_diffusivity[:3], face_concentrations[:, :3])
            numerator = -np.einsum(
                "i,...i->...", charge_weighted_diffusivity, concentration_gradient[:, :3]
            )
            # A completely empty face also has a zero concentration gradient.
            # Defining its field as zero avoids a removable 0/0 there.
            electric_gradient = np.divide(numerator, denom, out=np.zeros_like(numerator), where=denom > 0.0)

            ion_flux = -species_diffusivity[:2] * (
                concentration_gradient[:, :2] + face_concentrations[:, :2] * electric_gradient[:, None]
            )
            cluster_flux = -species_diffusivity[3] * concentration_gradient[:, 3]
            constituent_flux = ion_flux + cluster_flux[:, None]
            flux = np.zeros((n + 1, 2), dtype=float)
            flux[1:-1] = constituent_flux
            return (-np.diff(flux, axis=0) / widths[:, None]).ravel()

        # A row depends only on its own and the immediately neighbouring
        # two-component cells.
        state_size = 2 * n
        sparsity = diags(
            [np.ones(state_size - abs(offset)) for offset in range(-3, 4)],
            range(-3, 4),
            shape=(state_size, state_size),
            format="csr",
        )
        result = solve_ivp(
            rhs,
            (0.0, float(t[-1])),
            initial.ravel(),
            method="BDF",
            t_eval=t,
            rtol=2e-8,
            atol=2e-10,
            jac_sparsity=sparsity,
        )
        if not result.success:
            raise RuntimeError(result.message)
        prediction = result.y.T.reshape(len(t), n, 2)
        # The integrator's t=0 value is normally bit-for-bit identical; set it
        # explicitly to honour the preparation input exactly.
        if t[0] == 0.0:
            prediction[0] = initial
        return prediction

    def correlate(self, t, totals, harmonic=1):
        """Return normalized equilibrium two-time covariances for one mode."""
        diffusivity = self._require_diffusivity()
        lag = np.asarray(t, dtype=float)
        totals = np.asarray(totals, dtype=float)
        if lag.ndim != 1 or not np.isfinite(lag).all() or np.any(lag < 0.0) or np.any(np.diff(lag) < 0.0):
            raise ValueError("t must be an increasing one-dimensional array of non-negative lags")
        if totals.shape != (2,) or not np.isfinite(totals).all() or np.any(totals < 0.2) or np.any(totals > 2.0):
            raise ValueError("totals must contain two concentrations in [0.2, 2]")
        if isinstance(harmonic, (bool, np.bool_)) or int(harmonic) != harmonic or not 1 <= int(harmonic) <= 8:
            raise ValueError("harmonic must be an integer from 1 through 8")

        susceptibility, mobility = fluctuation_matrices(totals, diffusivity)
        wave_number = int(harmonic) * np.pi / CELL_LENGTH
        relaxation = wave_number**2 * np.linalg.solve(susceptibility.T, mobility.T).T

        # Per the specified predictor, the two conserved constituent counts
        # have independent ideal fluctuations at zero lag.
        initial_covariance = np.diag(totals)
        if len(lag) == 0:
            return np.empty((0, 2, 2), dtype=float)
        return np.stack([expm(-float(time) * relaxation) @ initial_covariance for time in lag], axis=0)


def fluctuation_matrices(totals: np.ndarray, diffusivity: float) -> tuple[np.ndarray, np.ndarray]:
    """Thermodynamic susceptibility and zero-current Onsager mobility."""
    concentration = equilibrium(np.asarray(totals, dtype=float)[None, :])[0]
    charge = np.array([1.0, 1.0, -1.0, 0.0])
    constituents = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]])

    population = np.diag(concentration)
    charge_variance = float(charge @ population @ charge)
    conditional = population - np.outer(population @ charge, population @ charge) / charge_variance
    susceptibility = constituents @ conditional @ constituents.T

    species_mobility = np.diag(diffusivity * np.array([1.0, 4.0, 2.0, NEUTRAL_MOBILITY]) * concentration)
    mobile_charge = species_mobility @ charge
    constrained = species_mobility - np.outer(mobile_charge, mobile_charge) / float(charge @ species_mobility @ charge)
    mobility = constituents @ constrained @ constituents.T
    return susceptibility, mobility


def equilibrium_correlations(t, totals, harmonic, diffusivity):
    """Functional counterpart of :meth:`TransportModel.correlate`."""
    susceptibility, mobility = fluctuation_matrices(np.asarray(totals, dtype=float), float(diffusivity))
    relaxation = (int(harmonic) * np.pi / CELL_LENGTH) ** 2 * np.linalg.solve(susceptibility.T, mobility.T).T
    lag = np.asarray(t, dtype=float)
    if len(lag) == 0:
        return np.empty((0, 2, 2), dtype=float)
    return np.stack([expm(-float(time) * relaxation) @ np.diag(totals) for time in lag], axis=0)
