import numpy as np
from scipy.optimize import minimize_scalar


_STRENGTH_BOUNDS = (0.6, 1.0)


def _mean_transmissions(max_count, q):
    """Return the phase-averaged transmissions for counts 0..max_count.

    Put x = 1/T - 1 for a chain ending in a barrier.  If another identical
    barrier is added after a uniformly random propagation phase, then

        1 + x' = (1+x)(1+q^2) + x q^2
                + 2 q sqrt(x(1+x)(1+q^2)) cos(theta),

    where q=b/k and theta is uniform on [0, 2*pi).  The recurrence below
    applies this phase average by Gauss-Chebyshev quadrature.  The hyperbolic
    coordinate mu=asinh(sqrt(x)) keeps the numerical grid compact even for
    the largest count in the stated domain.
    """
    max_count = int(max_count)
    result = np.ones(max_count + 1, dtype=float)
    if max_count == 0:
        return result

    q = float(q)
    result[1] = 1.0 / (1.0 + q * q)
    if max_count == 1:
        return result

    # The largest possible mu after max_count barriers is max_count*gamma.
    # These fixed quadrature sizes make predictions deterministic while
    # keeping interpolation and phase-integration errors well below the
    # calibration noise level.
    gamma = np.arcsinh(q)
    grid = np.linspace(0.0, max_count * gamma, 4097)
    values = 1.0 / np.cosh(grid) ** 2

    phase_nodes = np.cos(np.pi * (np.arange(96) + 0.5) / 96.0)
    cosh_gamma = np.cosh(gamma)
    sinh_gamma = np.sinh(gamma)
    cosh_mu = np.cosh(grid)[:, None]
    sinh_mu = np.sinh(grid)[:, None]

    # values is g_m(mu): the expected final transmission after m further
    # barriers, starting at radial coordinate mu.  Its value at mu=gamma is
    # the ensemble mean for m+1 barriers.
    for count in range(1, max_count):
        z = (
            (cosh_mu * cosh_gamma) ** 2
            + (sinh_mu * sinh_gamma) ** 2
            + 2.0 * cosh_mu * cosh_gamma * sinh_mu * sinh_gamma * phase_nodes
        )
        next_mu = np.arccosh(np.sqrt(np.maximum(z, 1.0)))
        values = np.mean(np.interp(next_mu, grid, values), axis=1)
        result[count + 1] = np.interp(gamma, grid, values)

    # This also makes the exact two-barrier value explicit and avoids asking
    # the numerical quadrature to reproduce it during calibration.
    result[2] = 1.0 / (1.0 + 2.0 * q * q)
    return result


def response(e, strength):
    """Mean power transmission for one experiment."""
    count = int(e["count"])
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return 1.0

    wave_number = float(e["wave_number"])
    q = float(strength) / wave_number
    if count == 1:
        return 1.0 / (1.0 + q * q)
    if count == 2:
        return 1.0 / (1.0 + 2.0 * q * q)
    return float(_mean_transmissions(count, q)[count])


def predict_at(experiments, strength):
    """Predict in input order, sharing phase-averaging work by wave number."""
    experiments = list(experiments)
    predictions = np.empty(len(experiments), dtype=float)
    if strength is None:
        raise ValueError("fit the model before requesting predictions")

    groups = {}
    for index, experiment in enumerate(experiments):
        count = int(experiment["count"])
        if count < 0:
            raise ValueError("count must be non-negative")
        q = float(strength) / float(experiment["wave_number"])
        groups.setdefault(q, []).append((index, count))

    for q, members in groups.items():
        values = _mean_transmissions(max(count for _, count in members), q)
        for index, count in members:
            predictions[index] = values[count]
    return predictions


class Model:
    def __init__(self):
        self.strength = None

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        def objective(strength):
            total = 0.0
            for record in records:
                sigma = float(record["sigma"])
                residual = (
                    response(record["input"], strength) - float(record["value"])
                ) / sigma
                total += residual * residual
            return total

        fitted = minimize_scalar(
            objective,
            bounds=_STRENGTH_BOUNDS,
            method="bounded",
            options={"xatol": 1e-12},
        )
        self.strength = float(np.clip(fitted.x, *_STRENGTH_BOUNDS))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.strength)
