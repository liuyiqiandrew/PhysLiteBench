"""Fixed half-filled WASEP screen; no task data or model evaluations."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import signal
import time
import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import ellipk, ellipe
from scipy.signal import resample

ROOT = Path(__file__).resolve().parent


@lru_cache(None)
def quadrature(order):
    x, w = np.polynomial.legendre.leggauss(order)
    theta = np.pi*(x+1)/4
    return np.sin(theta)**2, w*np.pi/4


def branch_point(field, modulus_squared, order):
    m = float(modulus_squared)
    K, E = float(ellipk(m)), float(ellipe(m))
    b = 4*K/field
    a2 = m*b*b
    sine2, weights = quadrature(order)
    third = float(np.sum(weights/((1-a2*sine2)*np.sqrt(1-m*sine2))))
    H = third/K
    eta = field*np.sqrt(max(0., (1-a2)*(1-b*b)))*H
    psi = 4*K*K*(1-m-2*E/K)
    return dict(m=m, a2=a2, b=b, H=H, eta=float(eta), psi=float(psi))


def elliptic(field, bias, order):
    eta = abs(field+bias)
    flat = bias*(bias+2*field)/4
    if field <= 2*np.pi or eta*eta >= field*field-4*np.pi*np.pi-1e-12:
        return dict(psi=flat, branch='flat', residual=0.)
    upper = brentq(lambda m: 4*ellipk(m)-field, 0., 1-1e-12, xtol=5e-15)
    if eta == 0:
        m = upper
    else:
        m = brentq(lambda m: branch_point(field, m, order)['eta']-eta,
                   0., upper, xtol=5e-15)
    result = branch_point(field, m, order)
    result.update(branch='nonuniform', residual=abs(result['eta']-eta))
    return result


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


def main():
    output = ROOT/'report.json'
    assert not output.exists(), 'Never overwrite a scientific execution.'
    plan_path = ROOT/'plan.json'
    plan = json.loads(plan_path.read_text())
    started = time.monotonic()
    report = dict(status='running', elliptic=[], spatial=[], monotonicity=[],
                  anchors=[], comparisons=[], gates={})
    report['source_sha256'] = {
        str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), plan_path]}
    def save():
        report['seconds'] = time.monotonic()-started
        output.write_text(json.dumps(report, indent=2)+'\n')
    def time_limit(signum, frame):
        report['status'] = 'time_limit_unresolved'
        save()
        raise TimeoutError('Fixed 600-second scientific screen budget reached.')
    signal.signal(signal.SIGALRM, time_limit)
    signal.alarm(600)
    save()
    for field in plan['fields']:
        upper = brentq(lambda m: 4*ellipk(m)-field, 0., 1-1e-12, xtol=5e-15)
        rows = [branch_point(field, m, 128) for m in np.linspace(0., upper, 201)]
        eta = np.array([r['eta'] for r in rows])
        report['monotonicity'].append(dict(field=field, rows=rows,
            maximum_increase=float(np.max(np.diff(eta))),
            endpoint_residual=float(eta[-1])))
    settings = [(f, f*r, 'prediction') for f,r in plan['cases']]
    settings += [(f,b,'control') for f,b in plan['controls']]
    for field, bias, kind in settings:
        for order in plan['quadrature_orders']:
            report['elliptic'].append(dict(field=field, bias=bias, order=order, kind=kind,
                                          **elliptic(field, bias, order)))
        for n in plan['grids']:
            for seed_name in plan['starts']:
                row = optimize_profile(field, bias, n, seed_name)
                report['spatial'].append(row)
                print(json.dumps({k:v for k,v in row.items() if k!='density'}), flush=True)
                save()
    for field, bias in plan['exact_anchors']:
        value = elliptic(field, bias, 128)
        report['anchors'].append(dict(field=field, bias=bias, **value,
            absolute_error=abs(value['psi']-bias*(bias+2*field)/4)))
    for field, bias, kind in settings:
        rows = [r for r in report['elliptic'] if r['field']==field and r['bias']==bias]
        truth = rows[-1]['psi']
        best = []
        selected = []
        for n in plan['grids']:
            candidates = [r for r in report['spatial'] if r['field']==field
                          and r['bias']==bias and r['n']==n]
            # Fixed multistart optimization: retain every outcome and choose
            # the largest finite psi among converged constrained profiles.
            candidates = [r for r in candidates if r['success']
                          and np.isfinite(r['psi']) and np.isfinite(r['density']).all()
                          and r['mass_error']<1e-9 and r['bound_margin']>1e-5
                          and r['normalized_tangent_gradient']<1e-6]
            chosen = max(candidates, key=lambda r:r['psi']) if candidates else None
            best.append(chosen['psi'] if chosen else None)
            selected.append(chosen)
        reflected = elliptic(field, -2*field-bias, 128)['psi']
        normalization = max(1.,abs(truth))
        fine = selected[-1]
        report['comparisons'].append(dict(field=field, bias=bias, psi=truth, kind=kind,
            quadrature_relative=abs(rows[0]['psi']-truth)/normalization,
            fine_spatial_relative=abs(best[-1]-truth)/normalization if fine else None,
            spatial_refinement_relative=abs(best[-1]-best[0])/normalization if all(selected) else None,
            doubled_mesh_relative=abs(fine['doubled_psi']-fine['psi'])/normalization if fine else None,
            selected_seeds=[r['seed'] if r else None for r in selected],
            selected_constraints=[dict(mass=r['mass_error'], doubled_mass=r['doubled_mass_error'],
                gradient=r['normalized_tangent_gradient'], bound_margin=r['bound_margin'])
                if r else None for r in selected],
            profile_contrast=float(np.ptp(fine['density'])) if fine else None,
            best_spatial=best, symmetry_absolute=abs(reflected-truth),
            flat=bias*(bias+2*field)/4))
    groups = {}
    for field in plan['fields']:
        rows = [r for r in report['comparisons'] if r['field']==field and r['kind']=='prediction']
        truth = np.array([r['psi'] for r in rows])
        flat = np.array([r['flat'] for r in rows])
        groups[str(field)] = float(np.linalg.norm(flat-truth)/np.linalg.norm(truth))
    report['group_source_errors'] = groups
    report['gates'] = dict(
        all_spatial_runs_completed=len(report['spatial'])==72,
        constrained_interior=all(c is not None and c['mass']<1e-9 and c['bound_margin']>1e-5
                                and c['gradient']<1e-6 and c['doubled_mass']<1e-9
                                for r in report['comparisons'] for c in r['selected_constraints']),
        quadrature=all(r['quadrature_relative']<1e-7 for r in report['comparisons']),
        branch_residual=all(r['residual']<1e-5 for r in report['elliptic']),
        independent_spatial=all(r['fine_spatial_relative'] is not None and r['fine_spatial_relative']<.002
                                for r in report['comparisons']),
        spatial_refinement=all(r['spatial_refinement_relative'] is not None and r['spatial_refinement_relative']<.002
                               for r in report['comparisons']),
        doubled_mesh=all(r['doubled_mesh_relative'] is not None and r['doubled_mesh_relative']<.002
                         for r in report['comparisons']),
        prediction_contrast=all(r['profile_contrast'] is not None and r['profile_contrast']>.01
                                for r in report['comparisons'] if r['kind']=='prediction'),
        not_below_flat=all(r['psi']>=r['flat']-1e-6*max(1.,abs(r['psi']))
                           for r in report['comparisons']),
        symmetry=all(r['symmetry_absolute']<1e-10 for r in report['comparisons']),
        exact_anchors=all(r['absolute_error']<1e-12 for r in report['anchors']),
        monotonicity=all(r['maximum_increase']<1e-8 and r['endpoint_residual']<1e-5
                         for r in report['monotonicity']),
        separation=all(x>.06 for x in groups.values()))
    numerical = all(v for k,v in report['gates'].items() if k!='separation')
    report['status'] = ('numerical_unresolved' if not numerical else
                        'screen_pass' if report['gates']['separation'] else 'insufficient_separation')
    save()
    signal.alarm(0)
    print(json.dumps({k:report[k] for k in ['status','gates','group_source_errors','seconds']}, indent=2))


if __name__ == '__main__':
    main()
