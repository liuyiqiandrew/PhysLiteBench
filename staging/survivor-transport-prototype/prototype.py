"""Survival-selected axial transport: analytic radial modes versus annular PDE."""

import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq, minimize_scalar
from scipy.special import j0, j1, jn_zeros, roots_legendre


ROOT = Path(__file__).resolve().parent
JZERO = float(jn_zeros(0, 1)[0])
X, W = roots_legendre(192)
RAD = (X + 1) / 2
WEIGHT = W * RAD / 2


def mode(diffusivity, radius, capture):
    biot = capture * radius / diffusivity
    if biot == 0:
        return 0.0
    return brentq(lambda q: q * j1(q) - biot * j0(q), 0, JZERO,
                  xtol=1e-14)


def analytic(diffusivity, radius, capture, plug, peak):
    q = mode(diffusivity, radius, capture)
    profile = j0(q * RAD)
    velocity = plug + peak * (1 - RAD**2)
    end_weight = WEIGHT * profile
    path_weight = end_weight * profile
    return {
        "loss_rate": diffusivity * q*q / radius**2,
        "source_drift": float(np.dot(end_weight, velocity) / end_weight.sum()),
        "physical_drift": float(np.dot(path_weight, velocity) / path_weight.sum()),
        "mode": q,
    }


def radial_operator(diffusivity, radius, capture, cells):
    """Finite-volume radial Laplacian with the physical Robin face flux."""
    dr = radius / cells
    edges = np.linspace(0, radius, cells + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    volume = (edges[1:]**2 - edges[:-1]**2) / 2
    coupling = diffusivity * edges[1:-1] / dr
    diagonal = np.zeros(cells)
    diagonal[:-1] += coupling
    diagonal[1:] += coupling
    diagonal[-1] += radius * capture / (1 + capture * dr / (2*diffusivity))
    # Similarity transform by sqrt(cell volume) makes the generator symmetric.
    diagonal = -diagonal / volume
    off_diagonal = coupling / np.sqrt(volume[:-1] * volume[1:])
    return centers, volume, diagonal, off_diagonal


def principal(diagonal, off_diagonal):
    n = len(diagonal)
    return float(eigh_tridiagonal(diagonal, off_diagonal, eigvals_only=True,
                                 select="i", select_range=(n-1, n-1))[0])


def tilted_reference(diffusivity, radius, capture, plug, peak, cells=512):
    r, vol, diagonal, off = radial_operator(diffusivity, radius, capture, cells)
    velocity = plug + peak * (1 - r*r/radius**2)
    lam0 = principal(diagonal, off)
    def derivative(h):
        plus = principal(diagonal + h*velocity + diffusivity*h*h, off)
        minus = principal(diagonal - h*velocity + diffusivity*h*h, off)
        return (plus-minus)/(2*h)
    # The tilted operator generates exp(s Z_t), including ordinary axial diffusion.
    # Derive its long-time first cumulant by eigenvalue differences, not phi² weights.
    d1, d2 = derivative(.08), derivative(.04)
    return {"loss_rate": -lam0, "drift": (4*d2-d1)/3}


def conditional_moment(diffusivity, radius, capture, plug, peak, times, cells=128):
    """Solve the coupled zeroth/first axial moment PDE by its modal propagator."""
    r, vol, diagonal, off = radial_operator(diffusivity, radius, capture, cells)
    eigen, basis = eigh_tridiagonal(diagonal, off)
    eigen -= eigen[-1]  # Cancel the common decay before conditional normalization.
    initial = basis.T @ np.sqrt(vol)  # Uniform transverse injection at Z=0.
    observe = initial.copy()
    velocity = plug + peak*(1-r*r/radius**2)
    coupling = basis.T @ (velocity[:, None] * basis)
    difference = eigen[None, :] - eigen[:, None]
    out = []
    for t in times:
        exponential = np.exp(eigen*t)
        numerator = exponential[None, :] - exponential[:, None]
        integral = np.divide(numerator, difference, out=np.zeros_like(difference),
                             where=np.abs(difference) > 1e-12)
        np.fill_diagonal(integral, t*exponential)
        survival = np.dot(observe * exponential, initial)
        first = np.sum(observe[:, None] * coupling * integral * initial[None, :])
        out.append(float(first/survival))
    return out


def main():
    started = time.perf_counter()
    domain = []
    for d, radius, capture, peak in itertools.product(
            [.7, 1.05, 1.4], [.7, 1.2], [.5, 2, 8, 24], [.6, -1.3]):
        a = analytic(d, radius, capture, 0, peak)
        coarse = tilted_reference(d, radius, capture, 0, peak, 256)
        fine = tilted_reference(d, radius, capture, 0, peak, 512)
        extrapolated = {k: (4*fine[k]-coarse[k])/3 for k in fine}
        domain.append({"diffusivity": d, "radius": radius, "capture": capture,
                       "peak": peak, **a, "reference": extrapolated,
                       "relative_drift_error": abs(extrapolated["drift"]-a["physical_drift"])/abs(a["physical_drift"]),
                       "source_relative_error": abs(a["source_drift"]-a["physical_drift"])/abs(a["physical_drift"])})
    hidden = []
    for radius, capture, peak in [(1, 8, 1), (.8, 20, -1.3), (1.2, 24, .7)]:
        a = analytic(1.05, radius, capture, 0, peak)
        coarse = tilted_reference(1.05, radius, capture, 0, peak, 512)
        fine = tilted_reference(1.05, radius, capture, 0, peak, 1024)
        gap = None
        r, v, diag, off = radial_operator(1.05, radius, capture, 128)
        ev = eigh_tridiagonal(diag, off, eigvals_only=True,
                             select="i", select_range=(126, 127))
        gap = float(ev[1]-ev[0])
        t0 = 2/gap
        times = [t0, 2*t0, 4*t0, 8*t0, 16*t0]
        moments = conditional_moment(1.05, radius, capture, 0, peak, times)
        slopes = [(moments[i+1]-moments[i])/(times[i+1]-times[i]) for i in range(4)]
        hidden.append({"radius": radius, "capture": capture, "peak": peak, **a,
                       "reference_512": coarse, "reference_1024": fine,
                       "radial_gap": gap, "times": times, "conditional_mean": moments,
                       "successive_interval_slopes": slopes,
                       "last_slope_relative_error": abs(slopes[-1]/a["physical_drift"]-1)})
    limits = []
    for d, radius, capture, plug, peak in itertools.product(
            [.7, 1.4], [.7, 1.2], [0, .5, 24], [-.9, .9], [0, .6]):
        a = analytic(d, radius, capture, plug, peak)
        reverse = analytic(d, radius, capture, -plug, -peak)
        limits.append({"reversal_error": abs(a["physical_drift"]+reverse["physical_drift"]),
                       "anchor_error": abs(a["physical_drift"]-a["source_drift"]) if capture==0 or peak==0 else None})
    # Loss rates identify D; plug-flow records are exact but do not independently identify it.
    cal = list(itertools.product([.7, 1., 1.3], [.5, 2., 8.]))
    def prediction(d):
        return np.array([analytic(d, r, k, 0, 0)["loss_rate"] for r, k in cal])
    recovery = []
    for d in np.linspace(.7, 1.4, 25):
        data = prediction(d)
        fit = minimize_scalar(lambda x: np.sum((prediction(x)-data)**2),
                              bounds=(.7, 1.4), method="bounded",
                              options={"xatol": 1e-12})
        recovery.append({"true": float(d), "fit": float(fit.x), "relative_error": abs(fit.x/d-1)})
    dgrid = np.linspace(.7, 1.4, 101)
    rate_grid = np.array([prediction(d) for d in dgrid])
    sensitivities = np.diff(rate_grid, axis=0)/np.diff(dgrid)[:, None]
    assert np.min(sensitivities) > 0
    assert max(x["relative_drift_error"] for x in domain) < 2e-6
    assert max(x["last_slope_relative_error"] for x in hidden) < 1e-4
    assert max(x["anchor_error"] or 0 for x in limits) < 1e-13
    assert max(x["relative_error"] for x in recovery) < 1e-6
    report = {
        "status": "prototype_only_no_agent_runs", "domain_cases": len(domain),
        "max_drift_reference_error": max(x["relative_drift_error"] for x in domain),
        "max_loss_reference_error": max(abs(x["reference"]["loss_rate"]/x["loss_rate"]-1) for x in domain),
        "hidden_source_relative_errors": [abs(x["source_drift"]/x["physical_drift"]-1) for x in hidden],
        "minimum_hidden_signal": min(abs(x["physical_drift"]) for x in hidden),
        "minimum_calibration_derivative": float(sensitivities.min()),
        "max_noiseless_parameter_error": max(x["relative_error"] for x in recovery),
        "max_reversal_error": max(x["reversal_error"] for x in limits),
        "max_anchor_error": max(x["anchor_error"] or 0 for x in limits),
        "domain": domain, "hidden": hidden, "parameter_recovery": recovery,
        "seconds": time.perf_counter()-started,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    path = ROOT / "report.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in {"domain","hidden","parameter_recovery"}}, indent=2))


if __name__ == "__main__":
    main()
