"""Bounded feasibility: current through deforming shorted material electrodes."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import brentq, minimize_scalar, root

HERE = Path(__file__).resolve().parent
S0 = np.array([.08, .06, .22])
GAMMA = 4.0


def drive(e):
    return np.array([.6*e[0, 2]+.2*e[0, 1],
                     .5*e[1, 2]-.1*e[0, 1],
                     .45*e[0, 0]+.30*e[1, 1]+.70*e[2, 2]+.20*e[0, 1]])


def energy(x, f, stiffness):
    e = (f.T@f-np.eye(3))/2
    return stiffness*np.dot(x, x)/2+GAMMA*np.dot(x, x)**2/4-np.dot(drive(e), x)


def state(f, stiffness):
    h = drive((f.T@f-np.eye(3))/2)
    hnorm = np.linalg.norm(h)
    if hnorm == 0:
        x = np.zeros(3)
    else:
        radius = brentq(lambda r: stiffness*r+GAMMA*r**3-hnorm, 0, hnorm/stiffness,
                        xtol=1e-15)
        x = radius*h/hnorm
    return S0+x


def response(f, g, stiffness):
    s = state(f, stiffness)
    x = s-S0
    tangent = (stiffness+GAMMA*np.dot(x, x))*np.eye(3)+2*GAMMA*np.outer(x, x)
    ds = np.linalg.solve(tangent, drive((f.T@g+g.T@f)/2))
    volume = np.linalg.det(f)
    polarization = -f@s/volume
    dpolarization = -(g@s+f@ds)/volume+f@s/volume*np.trace(np.linalg.solve(f, g))
    # Laboratory polarization-density derivative through the instantaneous face.
    source = -volume*np.linalg.solve(f.T, np.array([0., 0., 1.]))@dpolarization
    # Electrode charge response, derived independently below from electrostatics.
    physical = ds[2]
    return float(source), float(physical), s, polarization, dpolarization


def equilibrium_reference(f, stiffness):
    h = drive((f.T@f-np.eye(3))/2)
    result = root(lambda x: stiffness*x+GAMMA*np.dot(x, x)*x-h,
                  h/stiffness, tol=1e-12)
    residual = np.linalg.norm(stiffness*result.x+GAMMA*np.dot(result.x, result.x)*result.x-h)
    if residual > 1e-11:
        raise RuntimeError((result.message, residual))
    return S0+result.x


def electrode_charge(f, stiffness, cells=9, grid=2048):
    """Grounded Poisson solve for planar averages of all actual ion charges.

    q=+1 at fractional z=.2 and q=-1 at z=.2+s_z in every cell.
    The positive slab metric factor cancels when potential is converted back
    to total electrode charge, so solve in material z units. Both Dirichlet
    boundary potentials are zero. No polarization-current formula is used.
    """
    s = equilibrium_reference(f, stiffness)
    if not 0 < .2+s[2] < 1:
        raise ValueError('ion leaves selected material termination')
    spacing = cells/grid
    charge = np.zeros(grid+1)
    for j in range(cells):
        for position, value in ((j+.2, 1.), (j+.2+s[2], -1.)):
            index = position/spacing
            lower = int(np.floor(index))
            fraction = index-lower
            charge[lower] += value*(1-fraction)
            charge[lower+1] += value*fraction
    band = np.zeros((3, grid-1))
    band[0, 1:] = -1
    band[1] = 2
    band[2, :-1] = -1
    potential = solve_banded((1, 1), band, spacing*charge[1:-1])
    return float(-potential[-1]/spacing)


def reference(f, g, stiffness, step=2e-4, cells=9, grid=2048):
    return (electrode_charge(f-2*step*g, stiffness, cells, grid)
            -8*electrode_charge(f-step*g, stiffness, cells, grid)
            +8*electrode_charge(f+step*g, stiffness, cells, grid)
            -electrode_charge(f+2*step*g, stiffness, cells, grid))/(12*step)


def deformation(stretches, shears):
    f = np.diag(stretches).astype(float)
    f[0, 1], f[0, 2], f[1, 2] = shears
    return f


def calibration(stiffness):
    rows = []
    for extension in np.linspace(.88, 1.12, 9):
        for sign in (-1., 1.):
            f = np.diag([1., 1., extension])
            g = np.diag([0., 0., sign])
            rows.append(response(f, g, stiffness)[:2])
    return np.array(rows)


def main():
    start = time.perf_counter()
    rng = np.random.default_rng(428701)
    cases = []
    for stiffness in (.8, 1., 1.2):
        for stretch in (.88, 1., 1.12):
            for shear in (-.12, 0., .12):
                for g in (np.diag([1., 0., 0.]), np.diag([0., 1., 0.]),
                          np.array([[.8, .3, .2], [0, .4, -.1], [0, 0, .2]])):
                    cases.append((deformation([stretch, 1., 2-stretch], [shear, -shear/2, shear]), g, stiffness))
    for _ in range(64):
        f = deformation(rng.uniform(.88, 1.12, 3), rng.uniform(-.12, .12, 3))
        g = np.triu(rng.uniform(-1., 1., (3, 3)))
        cases.append((f, g, rng.uniform(.8, 1.2)))
    rows = []
    max_reference = max_refinement = max_cells = max_covariance = 0.
    charge_errors = []
    for f, g, stiffness in cases:
        source, exact, s, _, _ = response(f, g, stiffness)
        ref = reference(f, g, stiffness)
        refined = reference(f, g, stiffness, step=1e-4, grid=4096)
        alternate_cells = reference(f, g, stiffness, cells=15, grid=4096)
        max_reference = max(max_reference, abs(ref-exact))
        max_refinement = max(max_refinement, abs(ref-refined))
        max_cells = max(max_cells, abs(ref-alternate_cells))
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        rotated = response(q@f, q@g, stiffness)
        max_covariance = max(max_covariance, abs(rotated[0]-source), abs(rotated[1]-exact))
        charge_errors.append(abs(electrode_charge(f, stiffness)-s[2]))
        rows.append({'stiffness': stiffness, 'F': f.tolist(), 'G': g.tolist(),
                     'source': source, 'physical': exact, 'reference': ref,
                     'fractional_offset': s.tolist()})
    gaps = []
    for stiffness in np.linspace(.8, 1.2, 41):
        for stretch in (.92, 1., 1.08):
            for direction in (0, 1):
                f = deformation([stretch, 1., 1.04], [.05, -.04, .03])
                g = np.zeros((3, 3)); g[direction, direction] = 1
                source, exact, *_ = response(f, g, stiffness)
                gaps.append({'stiffness':float(stiffness), 'stretch':stretch, 'direction':direction,
                             'source':source, 'physical':exact, 'relative_gap':abs(source-exact)/abs(exact)})
    fit_rows = []
    for stiffness in np.linspace(.8, 1.2, 41):
        target = calibration(stiffness)[:, 1]
        fit = minimize_scalar(lambda k: np.mean((calibration(k)[:, 0]-target)**2),
                              bounds=(.8, 1.2), method='bounded', options={'xatol':1e-13})
        choices = [(fit.fun, fit.x), (np.mean((calibration(.8)[:, 0]-target)**2), .8),
                   (np.mean((calibration(1.2)[:, 0]-target)**2), 1.2)]
        recovered = min(choices)[1]
        fit_rows.append({'true':float(stiffness), 'fit':float(recovered)})
    cal_equality = max(float(np.max(np.abs(calibration(k)[:, 0]-calibration(k)[:, 1])))
                       for k in np.linspace(.8, 1.2, 41))
    # Frozen fractional charges under rigid rotation: electrode charge is constant.
    f = deformation([1.06, .95, 1.03], [.04, -.06, .03])
    omega = np.array([[0., 0., .6], [0., 0., -.3], [-.6, .3, 0.]])
    rotation = response(f, omega@f, 1.)[:2]
    report = {
        'status':'bounded_science_check_complete_not_task_or_model_evaluation',
        'runtime_seconds':time.perf_counter()-start,
        'cases':len(cases), 'structured_cases':81, 'random_cases':64,
        'independent_poisson_response_max_absolute_error':max_reference,
        'step_grid_refinement_max_absolute_change':max_refinement,
        'slab_cell_count_change_max_absolute':max_cells,
        'induced_charge_vs_offset_max_absolute':max(charge_errors),
        'frame_covariance_max_absolute':max_covariance,
        'calibration_equivalence_max_absolute':cal_equality,
        'calibration_global_fit_count':len(fit_rows),
        'calibration_fit_max_absolute_error':max(abs(r['true']-r['fit']) for r in fit_rows),
        'illustrative_held_out_count':len(gaps),
        'minimum_held_out_absolute_physical_signal':min(abs(r['physical']) for r in gaps),
        'maximum_held_out_absolute_physical_signal':max(abs(r['physical']) for r in gaps),
        'minimum_held_out_relative_gap':min(r['relative_gap'] for r in gaps),
        'maximum_held_out_relative_gap':max(r['relative_gap'] for r in gaps),
        'minimum_held_out_source_signal':min(r['source'] for r in gaps),
        'rigid_rotation_limit':{'source':rotation[0], 'physical':rotation[1],
                               'qualification':'Unscored identity exposing the same moving-face closure; no zero-response hidden proposal.'},
        'fit_rows':fit_rows, 'domain_rows':rows, 'held_out_rows':gaps,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (HERE/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('_rows')}, indent=2))

if __name__ == '__main__':
    main()
