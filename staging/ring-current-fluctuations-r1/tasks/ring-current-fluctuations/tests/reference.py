"""Frozen independent grid gold plus its self-contained reproduction code."""
from pathlib import Path
import json
import time
import numpy as np
from scipy.optimize import minimize
from scipy.signal import resample

TRUE_PARAMETER = .937
HERE = Path(__file__).resolve().parent


def hidden_cases():
    return json.loads((HERE/"data/hidden.json").read_text())


def hidden_inputs():
    return {name: [r["input"] for r in rows] for name, rows in hidden_cases().items()}


def predict(experiments, diffusivity):
    gold = {(r["input"]["field"], r["input"]["bias"]): r["unit_value"]
            for rows in hidden_cases().values() for r in rows}
    return diffusivity*np.asarray([gold[(e["field"], e["bias"])] for e in experiments])


def functional(phi, field, bias):
    """Independent periodic edge-gradient discretization, z=sin(phi)."""
    n = len(phi)
    sine, cosine = np.sin(phi), np.cos(phi)
    inverse = 1/cosine**2
    H = np.mean(inverse)
    eta2 = (field+bias)**2
    difference = np.roll(phi, -1)-phi
    cost = (field**2*np.mean(cosine**2)
            + n*n*np.mean(difference**2)-eta2/H)/4
    gradient = (-field**2*sine*cosine
                + n*n*(2*phi-np.roll(phi, 1)-np.roll(phi, -1))
                + eta2*inverse*np.tan(phi)/H**2)/(2*n)
    return float(cost), gradient

def optimize_profile(field, bias, n, seed_name):
    x = np.arange(n)/n
    if seed_name == 'flat':
        phi = np.zeros(n)
    elif seed_name == 'first_mode':
        phi = np.arcsin(.5*np.cos(2*np.pi*x))
    else:
        phi = np.arcsin(.4*np.cos(4*np.pi*x)+.1*np.sin(2*np.pi*x))
    constraint = {'type': 'eq', 'fun': lambda p: np.mean(np.sin(p)),
                  'jac': lambda p: np.cos(p)/n}
    started = time.monotonic()
    result = minimize(functional, phi, args=(field, bias), jac=True,
                      method='SLSQP', bounds=[(-np.pi/2+.0002, np.pi/2-.0002)]*n,
                      constraints=[constraint],
                      options={'ftol': 1e-11, 'maxiter': 2000})
    phi = result.x
    gradient = functional(phi, field, bias)[1]
    normal = np.cos(phi)/n
    tangent = gradient-normal*(gradient@normal)/(normal@normal)
    doubled = resample(phi, 2*n)
    return dict(field=field, bias=bias, n=n, seed=seed_name,
                success=bool(result.success), message=str(result.message),
                iterations=int(result.nit), psi=float(-result.fun),
                mass_error=float(abs(np.mean(np.sin(phi)))),
                normalized_tangent_gradient=float(n*np.max(abs(tangent))/max(1.,field**2)),
                bound_margin=float(np.pi/2-.0002-np.max(abs(phi))),
                doubled_psi=-functional(doubled, field, bias)[0],
                doubled_mass_error=float(abs(np.mean(np.sin(doubled)))),
                density=((1+np.sin(phi))/2).tolist(),
                seconds=time.monotonic()-started)

def state(phi, field, bias):
    cost, gradient = functional(phi, field, bias)
    normal = np.cos(phi) / len(phi)
    multiplier = -float(gradient @ normal) / float(normal @ normal)
    residual = gradient + multiplier * normal
    stationarity = len(phi) * float(np.max(np.abs(residual))) / max(1., field**2)
    mass = float(np.mean(np.sin(phi)))
    return cost, gradient, normal, multiplier, residual, stationarity, mass

def cost_hessian(phi, field, bias):
    n = len(phi)
    inverse = 1 / np.cos(phi)**2
    tangent = np.tan(phi)
    H = float(np.mean(inverse))
    eta2 = (field + bias)**2
    w = inverse * tangent
    laplacian = 2*np.eye(n) - np.roll(np.eye(n), 1, axis=1) - np.roll(np.eye(n), -1, axis=1)
    diagonal = (-field**2*np.cos(2*phi)
                + eta2*inverse*(1 + 3*tangent**2)/H**2) / (2*n)
    return (n*laplacian/2 + np.diag(diagonal)
            - 2*eta2*np.outer(w, w)/(n*n*H**3))

def polish(original, plan):
    started = time.monotonic()
    field, bias, n = original['field'], original['bias'], original['n']
    phi = np.arcsin(2*np.asarray(original['density']) - 1)
    bound = np.pi/2 - plan['angle_bound_offset']
    target = plan['stationarity_target']
    mass_target = plan['mass_target']
    trace = []
    reason = 'iteration_limit'

    def merit(s):
        return max(s[5]/target, abs(s[6])/mass_target)

    for iteration in range(plan['maximum_newton_steps']):
        current = state(phi, field, bias)
        cost, gradient, normal, multiplier, residual, stationarity, mass = current
        trace.append(dict(iteration=iteration, psi=-cost,
                          stationarity=stationarity, mass_error=abs(mass)))
        if stationarity <= target and abs(mass) <= mass_target:
            reason = 'target_reached'
            break
        hessian = cost_hessian(phi, field, bias) + np.diag(-multiplier*np.sin(phi)/n)
        kkt = np.zeros((n+1, n+1))
        kkt[:n, :n] = hessian
        kkt[:n, n] = normal
        kkt[n, :n] = normal
        rhs = -np.concatenate([residual, [mass]])
        try:
            update, _, rank, _ = np.linalg.lstsq(kkt, rhs, rcond=plan['kkt_rcond'])
        except np.linalg.LinAlgError:
            reason = 'linear_solve_failed'
            break
        trace[-1]['kkt_rank'] = int(rank)
        accepted = False
        for halving in range(plan['maximum_halvings'] + 1):
            step = 2.**(-halving)
            candidate = phi + step*update[:n]
            if not np.isfinite(candidate).all() or np.max(np.abs(candidate)) >= bound:
                continue
            trial = state(candidate, field, bias)
            if not np.isfinite([trial[0], trial[5], trial[6]]).all():
                continue
            reached = trial[5] <= target and abs(trial[6]) <= mass_target
            if reached or merit(trial) <= (1-plan['residual_armijo']*step)*merit(current):
                phi = candidate
                trace[-1].update(step=step, halvings=halving)
                accepted = True
                break
        if not accepted:
            reason = 'residual_line_search_failed'
            break

    final = state(phi, field, bias)
    if final[5] <= target and abs(final[6]) <= mass_target:
        reason = 'target_reached'
    doubled = resample(phi, 2*n)
    row = dict(original)
    row.update(psi=-final[0], mass_error=abs(final[6]),
               normalized_tangent_gradient=final[5],
               bound_margin=float(bound-np.max(np.abs(phi))),
               doubled_psi=-functional(doubled, field, bias)[0],
               doubled_mass_error=float(abs(np.mean(np.sin(doubled)))),
               density=((1+np.sin(phi))/2).tolist(),
               original_seconds=original['seconds'], seconds=time.monotonic()-started,
               original_psi=original['psi'],
               original_stationarity=original['normalized_tangent_gradient'],
               polish_status=reason, polish_trace=trace)
    # Keep the original SLSQP success flag as provenance. Selection still also
    # requires the independently recomputed original mass/interior/KKT gates.
    return row

POLISH_PLAN = dict(angle_bound_offset=.0002, stationarity_target=1e-8,
                   mass_target=1e-12, maximum_newton_steps=12, maximum_halvings=16,
                   kkt_rcond=1e-12, residual_armijo=.0001)


def reproduce():
    """Reproduce six fixed predictions at both retained grids; no elliptic calls."""
    raw, refined, comparisons = [], [], []
    for name, cases in hidden_cases().items():
        if name == "exact_limits":
            continue
        for case in cases:
            field, bias = case["input"]["field"], case["input"]["bias"]
            selected = []
            for n in [128, 256]:
                candidates = []
                for seed in ["flat", "first_mode", "mixed_modes"]:
                    start = optimize_profile(field, bias, n, seed)
                    raw.append(start)
                    row = polish(start, POLISH_PLAN)
                    row["parent_row_index"] = len(raw)-1
                    refined.append(row)
                    if (row["success"] and np.isfinite(row["psi"])
                            and row["mass_error"] < 1e-9 and row["bound_margin"] > 1e-5
                            and row["normalized_tangent_gradient"] < 1e-6):
                        candidates.append(row)
                selected.append(max(candidates, key=lambda r: r["psi"]) if candidates else None)
            comparisons.append(dict(input=case["input"], expected=case["unit_value"],
                values=[r["psi"] if r else None for r in selected],
                selected_seeds=[r["seed"] if r else None for r in selected]))
    return dict(raw=raw, refined=refined, comparisons=comparisons)
