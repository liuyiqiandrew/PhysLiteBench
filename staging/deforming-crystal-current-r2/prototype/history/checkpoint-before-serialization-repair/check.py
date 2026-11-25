"""Bounded capacitor-loaded crystal check; not a task or model evaluation."""
from datetime import datetime, timezone
from itertools import product
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from scipy.optimize import brentq, minimize_scalar, root

HERE = Path(__file__).resolve().parent
S0 = np.array([.08, .06, .22])
E3 = np.array([0., 0., 1.])
SIGMA = .001


def coupling(e):
    return np.array([.60*e[0, 2]+.20*e[0, 1],
                     .50*e[1, 2]-.10*e[0, 1],
                     .45*e[0, 0]+.30*e[1, 1]+.70*e[2, 2]+.20*e[0, 1]])


def geometry(f, drive):
    normal = np.linalg.solve(f.T, E3)
    volume = np.linalg.det(f)
    c = volume*np.dot(normal, normal)
    normal_prime = -np.linalg.solve(f.T, drive.T@normal)
    cp = c*np.trace(np.linalg.solve(f, drive))+2*volume*np.dot(normal, normal_prime)
    return c, cp


def zero_field_state(f, stiffness):
    h = coupling((f.T@f-np.eye(3))/2)
    norm = np.linalg.norm(h)
    if norm == 0:
        return S0.copy()
    radius = brentq(lambda r: stiffness*r+4*r**3-norm, 0, norm/stiffness, xtol=1e-15)
    return S0+radius*h/norm


def loaded_state(f, stiffness, load, electrical_coupling=1.):
    """Minimize the reduced convex energy, including macroscopic field storage."""
    c, _ = geometry(f, np.zeros((3, 3)))
    r = 0. if load is None else electrical_coupling/(c+load)
    h = coupling((f.T@f-np.eye(3))/2)
    s = zero_field_state(f, stiffness)
    for _ in range(40):
        x = s-S0
        gradient = stiffness*x+4*np.dot(x, x)*x-h+r*s[2]*E3
        if np.linalg.norm(gradient, np.inf) < 2e-14:
            return s
        hessian = (stiffness+4*np.dot(x, x))*np.eye(3)+8*np.outer(x, x)+r*np.outer(E3, E3)
        s -= np.linalg.solve(hessian, gradient)
    raise RuntimeError('Reduced internal state did not converge')


def response(f, drive, stiffness, load, physical=True, electrical_coupling=1.):
    c, cp = geometry(f, drive)
    s = loaded_state(f, stiffness, load, electrical_coupling) if physical else zero_field_state(f, stiffness)
    x = s-S0
    hessian = (stiffness+4*np.dot(x, x))*np.eye(3)+8*np.outer(x, x)
    rhs = coupling((f.T@drive+drive.T@f)/2)
    if load is None:
        p, pp = 1., 0.
    else:
        p, pp = load/(c+load), -load*cp/(c+load)**2
        if physical:
            r = electrical_coupling/(c+load)
            rp = -electrical_coupling*cp/(c+load)**2
            hessian += r*np.outer(E3, E3)
            rhs -= rp*s[2]*E3
    ds = np.linalg.solve(hessian, rhs)
    return float(p*ds[2]+pp*s[2]), s, float(p*s[2])


def field_reference(f, stiffness, load, electrical_coupling=1.):
    """Independent unknowns: offset, lab normal E, voltage, two face charges.

    E uses q0/(epsilon*a0^2); V uses q0*N/(epsilon*a0).
    No reduced capacitor divider, reduced energy, or forward tangent is used.
    The reference is the uniform macroscopic Maxwell problem; microscopic
    zero-field electrostatics are already part of the supplied cell energy.
    """
    material_normal = np.linalg.solve(f.T, np.array([0., 0., 1.]))
    b = np.linalg.norm(material_normal)
    normal = material_normal/b
    area = np.linalg.det(f)*b
    gap = 1/b
    e = (f.T@f-np.eye(3))/2
    h = np.array([.60*e[0, 2]+.20*e[0, 1],
                  .50*e[1, 2]-.10*e[0, 1],
                  .45*e[0, 0]+.30*e[1, 1]+.70*e[2, 2]+.20*e[0, 1]])

    def equations(y):
        s, electric, voltage, crystal_charge, load_charge = y[:3], *y[3:]
        x = s-np.array([.08, .06, .22])
        internal = stiffness*x+4*np.dot(x, x)*x-h
        internal += electrical_coupling*electric*(f.T@normal)
        return np.r_[internal,
                     crystal_charge-s[2]+area*electric,
                     voltage+gap*electric,
                     voltage if load is None else load_charge-load*voltage,
                     crystal_charge+load_charge]

    guess = np.r_[np.array([.08, .06, .22])+h/stiffness, 0., 0., .22, -.22]
    result = root(equations, guess, tol=1e-11)
    residual = float(np.max(np.abs(equations(result.x))))
    if residual > 3e-11:
        raise RuntimeError(('Field/node reference failed', result.message, residual))
    return result.x, residual


def reference_response(f, drive, stiffness, load, step=2e-4, electrical_coupling=1.):
    charges = [field_reference(f+i*step*drive, stiffness, load, electrical_coupling)[0][5]
               for i in (-2, -1, 1, 2)]
    return float((charges[0]-8*charges[1]+8*charges[2]-charges[3])/(12*step))


def deformation(diagonal, shear=(0., 0., 0.)):
    f = np.diag(diagonal).astype(float)
    f[0, 1], f[0, 2], f[1, 2] = shear
    return f


def calibration(stiffness):
    return np.array([response(np.diag([1., 1., stretch]), np.diag([0., 0., sign]),
                              stiffness, None, physical=False)[0]
                     for stretch in np.linspace(.88, 1.12, 9) for sign in (-1., 1.)])


def fit(values):
    means = np.asarray(values).reshape(-1, 18).mean(axis=0)
    def loss(k):
        return float(np.mean((calibration(k)-means)**2))
    optimum = minimize_scalar(loss, bounds=(.8, 1.2), method='bounded', options={'xatol':1e-13})
    return float(min((.8, float(optimum.x), 1.2), key=loss))


def loaded_groups():
    return {str(load):[(deformation([.96, 1.04, z], [.04, -.03, .02]),
                       np.diag([.15, .05, 1.]), load) for z in (.9, .97, 1.04, 1.1)]
            for load in (.5, 1., 2.)}


def group_values(stiffness, physical):
    return {name:np.array([response(f, d, stiffness, load, physical)[0] for f, d, load in rows])
            for name, rows in loaded_groups().items()}


def normalized_error(a, b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run = HERE/'runs'/stamp
    run.mkdir(parents=True)
    shutil.copy2(__file__, run/'check.py')
    start = time.perf_counter()
    report = {'status':'running_bounded_feasibility', 'run':str(run.relative_to(HERE)),
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'domain_rows':[], 'group_rows':[], 'calibration_rows':[], 'noise_rows':[],
              'limit_rows':[], 'scaling_rows':[]}
    try:
        rng = np.random.default_rng(613907)
        directions = []
        for i, j in zip(*np.triu_indices(3)):
            drive = np.zeros((3, 3)); drive[i, j] = 1.
            directions.append(drive)
        cases = []
        for choices in product((0, 1), repeat=6):
            f = deformation([(.88, 1.12)[i] for i in choices[:3]],
                            [(-.12, .12)[i] for i in choices[3:]])
            for k, load in product((.8, 1.2), (.5, 2.)):
                for drive in directions:
                    cases.append(('corner_basis', f, drive, k, load))
        for _ in range(96):
            cases.append(('interior', deformation(rng.uniform(.88, 1.12, 3), rng.uniform(-.12, .12, 3)),
                          np.triu(rng.uniform(-1., 1., (3, 3))), rng.uniform(.8, 1.2), rng.uniform(.5, 2.)))
        for index, (kind, f, drive, k, load) in enumerate(cases):
            physical, state, charge = response(f, drive, k, load)
            source = response(f, drive, k, load, False)[0]
            reference = reference_response(f, drive, k, load)
            field, residual = field_reference(f, k, load)
            row = {'kind':kind, 'F':f.tolist(), 'G':drive.tolist(), 'K':float(k), 'load':float(load),
                   'source':source, 'physical':physical, 'reference':reference,
                   'reference_error':abs(reference-physical), 'state':state.tolist(),
                   'state_error':float(np.max(abs(state-field[:3]))), 'charge_error':abs(charge-field[5]),
                   'node_charge':float(field[5]+field[6]), 'field_residual':residual,
                   'sign_error':abs(physical+response(f, -drive, k, load)[0]),
                   'cell_min':float(np.min(np.array([.25, .25, .20])+state)),
                   'cell_max':float(np.max(np.array([.25, .25, .20])+state))}
            if index % 8 == 0 or kind == 'interior':
                refined = reference_response(f, drive, k, load, step=1e-4)
                row['refined_error'] = abs(refined-physical)
                row['refinement_change'] = abs(refined-reference)
            report['domain_rows'].append(row)

        for k in np.linspace(.8, 1.2, 41):
            target = calibration(k)
            recovered = fit(np.tile(target, 16))
            equality = max(abs(response(np.diag([1., 1., z]), np.diag([0., 0., sign]), k, None)[0]
                               -v) for (z, sign), v in zip(product(np.linspace(.88, 1.12, 9), (-1., 1.)), target))
            report['calibration_rows'].append({'K':float(k), 'fit':recovered, 'error':abs(recovered-k),
                                               'equivalence_error':equality})
            for name, rows in loaded_groups().items():
                source, physical, refs = [], [], []
                for f, drive, load in rows:
                    source.append(response(f, drive, k, load, False)[0])
                    physical.append(response(f, drive, k, load)[0])
                    refs.append(reference_response(f, drive, k, load))
                source, physical, refs = map(np.array, (source, physical, refs))
                report['group_rows'].append({'K':float(k), 'load':name, 'source':source.tolist(),
                                            'physical':physical.tolist(), 'reference':refs.tolist(),
                                            'source_error':normalized_error(source, physical),
                                            'reference_error':float(np.max(abs(refs-physical)))})

        # Preserve both nominal noise sampling and a separate true-K sweep.
        for family, true_values in [('nominal', np.full(256, 1.06)), ('domain', np.linspace(.8, 1.2, 256))]:
            for index, true_k in enumerate(true_values):
                values = np.tile(calibration(true_k), 16)+rng.normal(0., SIGMA, 288)
                fitted = fit(values)
                chi2 = float(np.sum(((np.tile(calibration(fitted), 16)-values)/SIGMA)**2)/287)
                truth = group_values(true_k, True)
                op, sp = group_values(fitted, True), group_values(fitted, False)
                report['noise_rows'].append({'family':family, 'index':index, 'true_K':float(true_k),
                    'fit':fitted, 'parameter_error':abs(fitted/true_k-1), 'chi2':chi2,
                    'physical_errors':{n:normalized_error(op[n], truth[n]) for n in truth},
                    'source_errors':{n:normalized_error(sp[n], truth[n]) for n in truth}})

        f = deformation([1.04, .93, 1.1], [.06, -.04, .03])
        drive = np.array([[.15, -.2, .1], [0., -.1, .3], [0., 0., 1.]])
        for k in (.8, 1., 1.2):
            for load in (None, .5, 1., 2., 1e2, 1e4, 1e6):
                for g in (0., 1e-6, .1, 1.):
                    p, state, charge = response(f, drive, k, load, electrical_coupling=g)
                    source = response(f, drive, k, load, False)[0]
                    ref = reference_response(f, drive, k, load, electrical_coupling=g)
                    report['limit_rows'].append({'K':k, 'load':load, 'g':g, 'source':source,
                        'physical':p, 'difference':p-source, 'reference_error':abs(ref-p),
                        'outside_proposed_inputs':g != 1. or (load is not None and load > 2.)})

        # Dimensional energy and load scaling, including deliberately incorrect fixed-CL limit.
        epsilon, a0, q0 = 2.3, .7, 1.4
        e0 = q0*q0/(epsilon*a0)
        c, _ = geometry(f, drive)
        _, state, q = response(f, drive, 1., 1.)
        for n, lateral in product((8, 64, 512), (50, 200)):
            c0 = epsilon*a0*lateral/n
            actual_area = a0*a0*lateral*np.linalg.det(f)*np.linalg.norm(np.linalg.solve(f.T, E3))
            gap = n*a0/np.linalg.norm(np.linalg.solve(f.T, E3))
            cb, cl = epsilon*actual_area/gap, c0
            qc, polarization_charge = q0*lateral*q, q0*lateral*state[2]
            field_energy = (qc-polarization_charge)**2/(2*cb)+qc**2/(2*cl)
            intensive = field_energy/(n*lateral*e0)
            expected = state[2]**2/(2*(c+1.))
            report['scaling_rows'].append({'N':n, 'S':lateral, 'Cb':cb, 'CL':cl,
                'capacitance_error':abs(cb/c0-c), 'intensive_field_energy':intensive,
                'energy_error':abs(intensive-expected),
                'fixed_dimensional_CL_counterexample_cL':1.*n/(epsilon*a0*lateral)})

        rows = report['domain_rows']
        noise = report['noise_rows']
        report['summary'] = {
            'corner_response_cases':sum(r['kind']=='corner_basis' for r in rows),
            'interior_response_cases':sum(r['kind']=='interior' for r in rows),
            'reference_max_error':max(r['reference_error'] for r in rows),
            'refinement_max_change':max(r.get('refinement_change', 0.) for r in rows),
            'refined_max_error':max(r.get('refined_error', 0.) for r in rows),
            'state_max_error':max(r['state_error'] for r in rows),
            'charge_max_error':max(r['charge_error'] for r in rows),
            'node_max_error':max(abs(r['node_charge']) for r in rows),
            'field_max_residual':max(r['field_residual'] for r in rows),
            'sign_max_error':max(r['sign_error'] for r in rows),
            'cell_min':min(r['cell_min'] for r in rows), 'cell_max':max(r['cell_max'] for r in rows),
            'noiseless_fit_max_error':max(r['error'] for r in report['calibration_rows']),
            'calibration_equivalence_max_error':max(r['equivalence_error'] for r in report['calibration_rows']),
            'sampled_group_gap_min':min(r['source_error'] for r in report['group_rows']),
            'sampled_physical_signal_min':min(min(r['physical']) for r in report['group_rows']),
            'sampled_source_signal_min':min(min(r['source']) for r in report['group_rows']),
            'group_reference_max_error':max(r['reference_error'] for r in report['group_rows']),
            'noise_max_chi2':max(r['chi2'] for r in noise),
            'noise_max_parameter_error':max(r['parameter_error'] for r in noise),
            'noise_max_physical_error':max(max(r['physical_errors'].values()) for r in noise),
            'noise_min_source_error':min(min(r['source_errors'].values()) for r in noise),
            'noise_calibration_pass_count':sum(r['chi2'] <= 1.5 and r['parameter_error'] <= .03 for r in noise),
            'noise_physical_pass_count':sum(max(r['physical_errors'].values()) <= .04 for r in noise),
            'noise_source_all_groups_fail_count':sum(min(r['source_errors'].values()) > .04 for r in noise),
            'limit_reference_max_error':max(r['reference_error'] for r in report['limit_rows']),
            'short_or_g0_max_difference':max(abs(r['difference']) for r in report['limit_rows'] if r['load'] is None or r['g']==0),
            'dimensional_energy_max_error':max(r['energy_error'] for r in report['scaling_rows'])}
        report['status'] = 'bounded_science_complete_no_task_or_model_evaluation'
        checks = report['summary']
        assert checks['reference_max_error'] < 1e-7
        assert checks['refinement_max_change'] < 1e-7
        assert checks['cell_min'] > 0 and checks['cell_max'] < 1
        assert checks['noiseless_fit_max_error'] < 3e-8
        assert checks['calibration_equivalence_max_error'] < 1e-12
        assert checks['sampled_group_gap_min'] > .04
        assert checks['sampled_physical_signal_min'] > .05
        assert checks['sampled_source_signal_min'] > .05
        assert checks['noise_calibration_pass_count'] == len(noise)
        assert checks['noise_physical_pass_count'] == len(noise)
        assert checks['noise_source_all_groups_fail_count'] == len(noise)
        assert checks['limit_reference_max_error'] < 1e-7
        assert checks['short_or_g0_max_difference'] < 1e-12
        assert checks['dimensional_energy_max_error'] < 1e-13
    except Exception as error:
        report['status'] = 'failed_check_preserved'
        report['exception'] = {'type':type(error).__name__, 'message':str(error), 'traceback':traceback.format_exc()}
        raise
    finally:
        report['runtime_seconds'] = time.perf_counter()-start
        payload = json.dumps(report, indent=2, allow_nan=False)+'\n'
        (run/'report.json').write_text(payload)
        (HERE/'report.json').write_text(payload)
        print(json.dumps({k:v for k,v in report.items() if not k.endswith('_rows')}, indent=2))


if __name__ == '__main__':
    main()
