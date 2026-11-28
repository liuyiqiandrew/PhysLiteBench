"""McKean return law and the renewal equation for heat and time at sticking."""
import numpy as np
from scipy.linalg import solve
from scipy.integrate import trapezoid
from scipy.special import erf, roots_legendre


def density(z):
    # Density of the logarithm of the incoming / outgoing speed ratio.
    return 1.5 / np.pi * np.exp(2.5 * z - np.logaddexp(0, 3 * z))


def geometry(z, threshold, diffusivity):
    r = np.exp(z)
    gamma = 1 - r + r * r
    k = 3 * r / gamma
    cut = gamma / (diffusivity * threshold * (1 + r * r))
    return r, gamma, k, cut


def slow_probability(cut, k):
    return erf(np.sqrt((1 + k) * cut)) - np.exp(-cut) * np.sqrt((1 + k) / k) * erf(
        np.sqrt(k * cut)
    )


def slow(q, diffusivity):
    z = np.linspace(-16, 72, 1001)
    _, _, k, cut = geometry(z, q["threshold"], diffusivity)
    return trapezoid(density(z) * slow_probability(cut, k), z)


def precise_drift(fast, soft, threshold, diffusivity):
    """Resolve signs near criticality using the compact return-speed integral."""
    from decimal import Decimal, localcontext

    with localcontext() as context:
        context.prec = 55
        pi = Decimal("3.141592653589793238462643383279502884197169399375105821")
        f, s, h, d = [
            Decimal.from_float(float(x)) for x in (fast, soft, threshold, diffusivity)
        ]
        base = pi / Decimal(3).sqrt() + f.ln()
        if f == s:
            return float(base)

        def erf_ratio(x):
            # Integral from zero to one of exp(-x*t²) dt.
            term = total = Decimal(1)
            n = 0
            while abs(term) > Decimal("1e-52"):
                n += 1
                term *= -x / n
                total += term / (2 * n + 1)
            return total

        def integrand(v):
            # Pair R and 1/R, and set v=exp(-|log R|/2).
            p = v * v / (1 + v**4)
            a = (1 + 2 * p) / (d * h)
            b = 3 * p / (d * h)
            gate = 2 * (a / pi).sqrt() * (erf_ratio(a) - (b - a).exp() * erf_ratio(b))
            return 3 / pi * (1 + v**4) / (1 + v**6) * gate

        previous = [(integrand(Decimal(0)) + integrand(Decimal(1))) / 2]
        for level in range(1, 12):
            n = 2**level
            step = Decimal(1) / n
            odd = sum(integrand(j * step) for j in range(1, n, 2))
            row = [previous[0] / 2 + step * odd]
            for k in range(1, level + 1):
                row.append(row[-1] + (row[-1] - previous[k - 1]) / (4**k - 1))
            if abs(row[-1] - previous[-1]) < Decimal("1e-33"):
                break
            previous = row
        return float(base + row[-1] * (s / f).ln())


def log_drift(q, diffusivity):
    p = slow(q, diffusivity)
    fast, soft = q["restitution"]
    critical = np.exp(-np.pi / np.sqrt(3))
    estimate = np.log(fast / critical) + p * np.log(soft / fast)
    if abs(estimate) < 1e-10:
        return precise_drift(fast, soft, q["threshold"], diffusivity)
    return estimate


def signal(q, diffusivity, step=0.14, lower=-56, order=24):
    if q["mode"] == "selector":
        return float(slow(q, diffusivity))
    drift = log_drift(q, diffusivity)
    if drift >= 0 or q["mode"] == "sticking":
        return float(drift < 0)
    speed = q["launch"]
    clock, dose = q["clock"], q["dose"]
    if clock == dose == 0:
        return 1.0
    y = np.arange(lower, 3.8 + step * 0.5, step)
    u2 = np.exp(2 * y)
    lhs = np.eye(len(y))
    rhs = np.zeros(len(y))
    nodes, weights = roots_legendre(order)
    row = np.arange(len(y))[:, None]
    for branch, factor in enumerate(q["restitution"]):
        offsets = np.arange(
            int(np.floor((-14 + np.log(factor)) / step)),
            int(np.ceil((68 + np.log(factor)) / step)) + 1,
        )
        z = offsets * step - np.log(factor)
        r, g, k, cut = geometry(z, q["threshold"], diffusivity)
        # Integrate in sqrt(gamma*u²/(D*t)); the gate divides the integral.
        bound = np.sqrt(cut)
        lo, hi = (
            (bound, np.full_like(bound, 7.1))
            if branch == 0
            else (np.zeros_like(bound), bound)
        )
        zeta = lo[:, None] + (hi - lo)[:, None] * (nodes + 1) / 2
        x = zeta * zeta
        w = (
            weights
            * (hi - lo)[:, None]
            * zeta
            * np.exp(-x)
            * np.sqrt((1 + k) / k)[:, None]
            * erf(np.sqrt(k[:, None] * x))
        )
        rates = (
            clock * g[:, None] / (diffusivity * x)
            + 0.5 * dose * (1 - factor * factor) * r[:, None] ** 2
        )
        # Batched in log speed to bound the temporary exponential array.
        coefficient = np.empty((len(y), len(z)))
        for i in range(0, len(y), 24):
            coefficient[i : i + 24] = (
                np.exp(-u2[i : i + 24, None, None] * rates[None, :, :]) * w
            ).sum(axis=2)
        coefficient *= step * density(z)
        col = row + offsets
        inside = (col >= 0) & (col < len(y))
        ri, ci = np.where(inside)
        lhs[ri, col[ri, ci]] -= coefficient[ri, ci]
        rhs += (coefficient * (col < 0)).sum(axis=1)
    value = solve(lhs, rhs, check_finite=False)
    from scipy.interpolate import CubicSpline

    return float(CubicSpline(y, value)(np.log(speed)))
