"""One fixed KKT polish of the 72 retained profiles; never rerun the screen."""
from pathlib import Path
import hashlib
import json
import signal
import time

import numpy as np
from scipy.signal import resample

from screen import functional


ROOT = Path(__file__).resolve().parent


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


def evaluate(report, plan):
    gates = plan['gates']
    settings = [(f, f*r, 'prediction') for f, r in plan['cases']]
    settings += [(f, b, 'control') for f, b in plan['controls']]
    report['comparisons'] = []
    for field, bias, kind in settings:
        rows = [r for r in report['elliptic'] if r['field'] == field and r['bias'] == bias]
        truth = rows[-1]['psi']
        selected = []
        for n in plan['grids']:
            candidates = [r for r in report['spatial'] if r['field'] == field
                          and r['bias'] == bias and r['n'] == n and r['success']
                          and np.isfinite(r['psi']) and np.isfinite(r['density']).all()
                          and r['mass_error'] < gates['mass_absolute']
                          and r['bound_margin'] > gates['bound_margin']
                          and r['normalized_tangent_gradient'] < gates['normalized_projected_gradient']]
            selected.append(max(candidates, key=lambda r: r['psi']) if candidates else None)
        best = [r['psi'] if r else None for r in selected]
        fine = selected[-1]
        normalization = max(1., abs(truth))
        original = next(r for r in report['original_comparisons']
                        if r['field'] == field and r['bias'] == bias)
        report['comparisons'].append(dict(
            field=field, bias=bias, kind=kind, psi=truth,
            quadrature_relative=abs(rows[0]['psi']-truth)/normalization,
            fine_spatial_relative=abs(best[-1]-truth)/normalization if fine else None,
            spatial_refinement_relative=abs(best[-1]-best[0])/normalization if all(selected) else None,
            doubled_mesh_relative=abs(fine['doubled_psi']-fine['psi'])/normalization if fine else None,
            selected_seeds=[r['seed'] if r else None for r in selected],
            selected_constraints=[dict(mass=r['mass_error'], doubled_mass=r['doubled_mass_error'],
                                       gradient=r['normalized_tangent_gradient'], bound_margin=r['bound_margin'])
                                  if r else None for r in selected],
            profile_contrast=float(np.ptp(fine['density'])) if fine else None,
            best_spatial=best, symmetry_absolute=original['symmetry_absolute'],
            flat=bias*(bias+2*field)/4))
    groups = {}
    for field in plan['fields']:
        rows = [r for r in report['comparisons'] if r['field'] == field and r['kind'] == 'prediction']
        truth = np.array([r['psi'] for r in rows])
        flat = np.array([r['flat'] for r in rows])
        groups[str(field)] = float(np.linalg.norm(flat-truth)/np.linalg.norm(truth))
    report['group_source_errors'] = groups
    comparisons = report['comparisons']
    report['gates'] = dict(
        all_spatial_runs_completed=len(report['spatial']) == gates['expected_spatial_runs'],
        constrained_interior=all(c is not None and c['mass'] < gates['mass_absolute']
                                and c['bound_margin'] > gates['bound_margin']
                                and c['gradient'] < gates['normalized_projected_gradient']
                                and c['doubled_mass'] < gates['mass_absolute']
                                for r in comparisons for c in r['selected_constraints']),
        quadrature=all(r['quadrature_relative'] < gates['quadrature_relative'] for r in comparisons),
        branch_residual=all(r['residual'] < gates['branch_residual_absolute'] for r in report['elliptic']),
        independent_spatial=all(r['fine_spatial_relative'] is not None
                                and r['fine_spatial_relative'] < gates['fine_spatial_relative'] for r in comparisons),
        spatial_refinement=all(r['spatial_refinement_relative'] is not None
                               and r['spatial_refinement_relative'] < gates['spatial_refinement_relative'] for r in comparisons),
        doubled_mesh=all(r['doubled_mesh_relative'] is not None
                         and r['doubled_mesh_relative'] < gates['doubled_mesh_relative'] for r in comparisons),
        prediction_contrast=all(r['profile_contrast'] is not None
                                and r['profile_contrast'] > gates['minimum_prediction_profile_contrast']
                                for r in comparisons if r['kind'] == 'prediction'),
        not_below_flat=all(r['psi'] >= r['flat']-1e-6*max(1., abs(r['psi'])) for r in comparisons),
        symmetry=all(r['symmetry_absolute'] < gates['symmetry_absolute'] for r in comparisons),
        exact_anchors=all(r['absolute_error'] < gates['exact_anchor_absolute'] for r in report['anchors']),
        monotonicity=all(r['maximum_increase'] < gates['map_maximum_increase']
                         and r['endpoint_residual'] < gates['map_endpoint_residual'] for r in report['monotonicity']),
        separation=all(x > gates['minimum_complete_group_source_error'] for x in groups.values()))
    numerical = all(v for k, v in report['gates'].items() if k != 'separation')
    report['status'] = ('numerical_unresolved' if not numerical else
                        'polishing_pass' if report['gates']['separation'] else 'insufficient_separation')


def main():
    plan_path = ROOT/'polishing-plan.json'
    polish_plan = json.loads(plan_path.read_text())
    for name, expected in polish_plan['input_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    screen_plan = json.loads((ROOT/'plan.json').read_text())
    source = json.loads((ROOT/'report.json').read_text())
    assert len(source['spatial']) == 72
    output = ROOT/'polishing-report.json'
    assert not output.exists(), 'Never overwrite a scientific execution.'
    started = time.monotonic()
    report = dict(status='polishing_running', parent_status=source['status'],
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__), plan_path, ROOT/'screen.py', ROOT/'plan.json', ROOT/'report.json']},
                  elliptic=source['elliptic'], monotonicity=source['monotonicity'],
                  anchors=source['anchors'], original_comparisons=source['comparisons'],
                  spatial=[], comparisons=[], gates={})

    def save():
        report['seconds'] = time.monotonic()-started
        output.write_text(json.dumps(report, indent=2)+'\n')

    def time_limit(signum, frame):
        report['status'] = 'time_limit_unresolved'
        save()
        raise TimeoutError('Fixed polishing budget reached.')

    signal.signal(signal.SIGALRM, time_limit)
    signal.alarm(polish_plan['budget_seconds'])
    save()
    for index, original in enumerate(source['spatial']):
        row = polish(original, polish_plan)
        row['parent_row_index'] = index
        report['spatial'].append(row)
        print(json.dumps({k: row[k] for k in ['parent_row_index', 'field', 'bias', 'n', 'seed',
                                              'polish_status', 'psi', 'normalized_tangent_gradient']}), flush=True)
        save()
    evaluate(report, screen_plan)
    save()
    signal.alarm(0)
    print(json.dumps({k: report[k] for k in ['status', 'gates', 'group_source_errors', 'seconds']}, indent=2))


if __name__ == '__main__':
    main()
