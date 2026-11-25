"""Nematic boundary-work screen; no task data or model evaluation."""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_bvp


def radial_state(inner_angle, outer_angle, radius_ratio, tolerance=1e-10):
    length = np.log(radius_ratio)
    x = np.linspace(0.0, 1.0, 48)
    guess = np.vstack((inner_angle + (outer_angle-inner_angle)*x,
                       np.full_like(x, outer_angle-inner_angle)))
    sol = solve_bvp(
        lambda s, y: np.vstack((y[1], length**2*np.sin(y[0])*np.cos(y[0]))),
        lambda left, right: np.array([left[0]-inner_angle, right[0]-outer_angle]),
        x, guess, tol=tolerance, max_nodes=4096,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def torques(inner_radius, radius_ratio, inner_angle, outer_angle, modulus):
    sol = radial_state(inner_angle, outer_angle, radius_ratio)
    outer_radius = inner_radius*radius_ratio
    slope = sol.y[1, -1]/(outer_radius*np.log(radius_ratio))
    correct = modulus*slope
    shortcut = modulus*(slope + np.sin(outer_angle)*np.cos(outer_angle)/outer_radius)
    return correct, shortcut, sol


def minimized_energy(inner_angle, outer_angle, radius_ratio, modulus,
                     modes=32, saddle_ratio=0.5):
    """Independent Ritz minimization of bulk div/curl energy minus surface flux.

    The returned energy is per unit axial length. No radial differential equation
    or torque formula enters this reference.
    """
    length = np.log(radius_ratio)
    q, w = leggauss(6*modes+16)
    s, w = (q+1)*length/2, w*length/2
    frequencies = np.pi*np.arange(1, modes+1)/length
    basis = np.sin(np.outer(s, frequencies))
    derivative = np.cos(np.outer(s, frequencies))*frequencies
    base = inner_angle+(outer_angle-inner_angle)*s/length
    base_slope = (outer_angle-inner_angle)/length
    coefficients = np.zeros(modes)
    for iteration in range(16):
        theta = base+basis@coefficients
        theta_s = base_slope+derivative@coefficients
        sine, cosine = np.sin(theta), np.cos(theta)
        # At phi=0, the nonzero Cartesian derivatives are
        # dn_x/dx=cos(theta)*theta_s/r, dn_z/dx=-sin(theta)*theta_s/r,
        # dn_y/dy=sin(theta)/r. The r^2 Jacobian cancels those denominators.
        div_scaled = cosine*theta_s+sine
        curl_scaled = sine*theta_s
        energy = np.pi*modulus*np.dot(w, div_scaled**2+curl_scaled**2)
        energy -= 2*np.pi*saddle_ratio*modulus*(np.sin(outer_angle)**2-np.sin(inner_angle)**2)
        dtheta = 2*theta_s*np.cos(2*theta)+2*sine*cosine
        dslope = 2*theta_s+2*sine*cosine
        gradient = np.pi*modulus*(basis.T@(w*dtheta)+derivative.T@(w*dslope))
        h00 = -4*theta_s*np.sin(2*theta)+2*np.cos(2*theta)
        h01 = 2*np.cos(2*theta)
        hessian = np.pi*modulus*(basis.T@((w*h00)[:, None]*basis)
                    +basis.T@((w*h01)[:, None]*derivative)
                    +derivative.T@((w*h01)[:, None]*basis)
                    +2*derivative.T@(w[:, None]*derivative))
        step = np.linalg.solve(hessian, gradient)
        if np.max(np.abs(step)) < 2e-13:
            return float(energy), {
                "iterations": iteration+1,
                "gradient_max": float(np.max(np.abs(gradient))),
                "reduced_energy": float(np.pi*modulus*np.dot(w, theta_s**2+sine**2)),
                "smallest_hessian_eigenvalue": float(np.linalg.eigvalsh(hessian)[0]),
            }
        coefficients -= step
    raise RuntimeError("Ritz minimization did not converge")


def energy_torque(inner_radius, radius_ratio, inner_angle, outer_angle, modulus,
                  modes=32, saddle_ratio=0.5, step=2e-4):
    energies = [minimized_energy(inner_angle, outer_angle+j*step, radius_ratio,
                                modulus, modes, saddle_ratio)[0]
                for j in [-2, -1, 1, 2]]
    derivative = (energies[0]-8*energies[1]+8*energies[2]-energies[3])/(12*step)
    return derivative/(2*np.pi*inner_radius*radius_ratio)


def main():
    rng = np.random.default_rng(241101)
    cases = list(itertools.product([0.7, 1.3], [2.0, 4.0], [.05, .15], [.3, .7], [.8, 1.6]))
    cases += [(rng.uniform(.7, 1.3), rng.uniform(2, 4), rng.uniform(.05, .15),
               rng.uniform(.3, .7), rng.uniform(.8, 1.6)) for _ in range(32)]
    records = []
    for case in cases:
        correct, shortcut, sol = torques(*case)
        ref = energy_torque(*case)
        bulk_ref = energy_torque(*case, saddle_ratio=0)
        a, ratio, low, high, modulus = case
        energy, detail = minimized_energy(low, high, ratio, modulus)
        sample = sol.sol(np.linspace(0, 1, 301))
        sample[1] /= np.log(ratio)
        records.append({
            "input": {"inner_radius": a, "radius_ratio": ratio,
                      "inner_angle": low, "outer_angle": high, "modulus": modulus},
            "oracle": correct, "shortcut": shortcut, "energy_reference": ref,
            "oracle_reference_relative_error": abs(correct-ref)/abs(correct),
            "shortcut_energy_relative_error": abs(shortcut-bulk_ref)/abs(shortcut),
            "shortcut_relative_error": abs(shortcut/correct-1),
            "full_energy": energy,
            "full_vs_gradient_energy_error": abs(energy-detail["reduced_energy"]),
            "second_variation_potential_max": float(np.max(sample[1]**2+np.sin(sample[0])**2)),
            "radial_poincare_constant": float(np.pi**2/np.log(ratio)**2),
            "profile_min": float(sample[0].min()),
            "profile_max": float(sample[0].max()),
        })
    refinements = []
    for case in cases[::12]:
        coarse = energy_torque(*case, modes=32)
        fine = energy_torque(*case, modes=64, step=1e-4)
        refinements.append(abs(coarse-fine)/abs(fine))

    # Flat slab energy has zero boundary divergence and a linear minimizer.
    calibration = []
    for thickness, low, high, modulus in itertools.product([.7, 1.3], [.05, .15], [.3, .7], [.8, 1.6]):
        correct = modulus*(high-low)/thickness
        e = lambda beta: .5*modulus*(beta-low)**2/thickness
        h = 1e-4
        reference = (e(high+h)-e(high-h))/(2*h)
        calibration.append(abs(correct-reference))

    largest_gradient_bound = (.65/np.log(2)+np.log(4)/2)**2+np.sin(.7)**2
    stability_gap = np.pi**2/np.log(4)**2-largest_gradient_bound
    # Planar fixed-gap limit: both curved torques approach K*(beta-alpha)/gap.
    plane_limits = []
    for a in [10, 100, 1000, 10000]:
        exact, wrong, _ = torques(a, 1+1/a, .1, .5, 1.2)
        plane_limits.append({"inner_radius": a, "oracle": exact, "shortcut": wrong,
                             "planar": .48, "relative_difference": abs(wrong/exact-1)})

    report = {
        "status": "prototype_only_not_evaluated",
        "mechanism": "Saddle-splay changes boundary actuator work while leaving a fixed-boundary bulk profile unchanged.",
        "energy_convention": "f=K*((div n)^2+|curl n|^2)/2-K24*div[n div n+n cross curl n]; K24=K/2",
        "readout": "External reversible work per outer-wall area per increase in the imposed radial-axial tilt; local torque axis e_phi, not net cylinder torque.",
        "domain": {"inner_radius": [.7, 1.3], "radius_ratio": [2, 4],
                   "inner_angle": [.05, .15], "outer_angle": [.3, .7], "modulus": [.8, 1.6]},
        "seed": 241101,
        "case_count": len(records),
        "max_oracle_reference_relative_error": max(x["oracle_reference_relative_error"] for x in records),
        "max_shortcut_own_energy_relative_error": max(x["shortcut_energy_relative_error"] for x in records),
        "shortcut_relative_error_range": [min(x["shortcut_relative_error"] for x in records), max(x["shortcut_relative_error"] for x in records)],
        "signal_range": [min(x["oracle"] for x in records), max(x["oracle"] for x in records)],
        "max_reference_refinement_relative_error": max(refinements),
        "max_full_vs_gradient_energy_error": max(x["full_vs_gradient_energy_error"] for x in records),
        "max_flat_calibration_error": max(calibration),
        "stability": {"global_gradient_density_nonnegative": True,
                      "meridional_convexity_min_cos_2theta": float(np.cos(1.4)),
                      "conservative_potential_bound": float(largest_gradient_bound),
                      "all_direction_dirichlet_second_variation_lower_bound": float(stability_gap),
                      "proof": "For any Cartesian vector perturbation v with zero boundary trace, radial Poincare in s=log(r/a) bounds integral |v_s|^2 by pi^2/log(4)^2 integral |v|^2. The director gradient potential r^2|grad n|^2=theta_s^2+sin(theta)^2 is bounded by (.65/log2+log4/2)^2+sin(.7)^2. Angular/axial gradient terms are nonnegative."},
        "flat_limit": plane_limits,
        "cases": records,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    assert report["max_oracle_reference_relative_error"] < 2e-6
    assert report["max_shortcut_own_energy_relative_error"] < 2e-6
    assert report["max_reference_refinement_relative_error"] < 2e-6
    assert report["shortcut_relative_error_range"][0] > .2
    assert stability_gap > 0
    Path(__file__).with_name("report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
