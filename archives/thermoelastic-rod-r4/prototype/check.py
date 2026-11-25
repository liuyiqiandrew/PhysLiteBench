"""Bounded standard-linear-solid caloric-closure study; no task or model calls."""
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eig, solve

L, AREA, T0, CE, CB = .05, 1e-4, 300., 1e6, 6.
E0, E1, STRAIN_SCALE = 2e9, 2e9, .002
TIMES = np.array([0., .15, 1., 5., 20., 80.])
PREPS = [(0., .8, 0., 0.), (.2, 0., .8, -.6), (-.3, .3, .4, .8)]


def geometry(n, k, contact, chi):
    dx = L / n
    x = (np.arange(n) + .5) * dx
    beta0 = E0 * .002 * (1 + .65 * np.cos(2 * np.pi * x / L))
    beta1 = chi * beta0
    eta = E1 * 12 * (1 + .97 * np.cos(2 * np.pi * x / L))
    lap = np.diag(np.r_[1., np.full(n - 2, 2.), 1.])
    lap -= np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    lap *= k * AREA / dx
    h = 0. if contact == 0 else 1 / (1 / contact + dx / (2 * k * AREA))
    heat = np.zeros((n + 1, n + 1))
    heat[:n, :n] = lap
    heat[0, 0] += h
    heat[-1, -1] = h
    heat[0, -1] = heat[-1, 0] = -h
    return x, beta0, beta1, eta, heat, AREA * dx


def forward(n, k, contact, chi, relaxed=False):
    x, b0, b1, eta, heat, volume = geometry(n, k, contact, chi)
    identity = np.eye(n)
    project = identity - np.ones((n, n)) / n
    bt = b0 + b1
    strain_t = project * bt[None, :] / (E0 + E1)
    strain_z = E1 / (E0 + E1) * project
    z_t = (E1 * strain_t - np.diag(b1)) / eta[:, None]
    z_z = (E1 * strain_z - E1 * identity) / eta[:, None]
    if relaxed:
        capacity = np.diag(CE + T0 * b1**2 / E1) + T0 * b0[:, None] * strain_t
        storage = T0 * b0[:, None] * strain_z
    else:
        capacity = CE * identity + T0 * bt[:, None] * strain_t
        storage = T0 * (bt[:, None] * strain_z - np.diag(b1))
    generator = np.zeros((2*n+1, 2*n+1))
    rhs = np.zeros((n, 2*n+1))
    rhs[:, :n] = -heat[:n, :n] / volume - storage @ z_t
    rhs[:, n:2*n] = -storage @ z_z
    rhs[:, -1] = -heat[:n, -1] / volume
    generator[:n] = solve(capacity, rhs)
    generator[n:2*n, :n], generator[n:2*n, n:2*n] = z_t, z_z
    generator[-1, :n] = -heat[-1, :n] / CB
    generator[-1, -1] = -heat[-1, -1] / CB
    scale = np.r_[np.ones(n), np.full(n, STRAIN_SCALE), 1.]
    generator = generator * scale[None, :] / scale[:, None]
    return generator, (x, b0, b1, eta, heat, volume, strain_t, strain_z)


def entropy_reference(n, k, contact, chi):
    """Solve common stress in entropy coordinates, then assemble balance laws."""
    x, b0, b1, eta, heat, volume = geometry(n, k, contact, chi)
    bt = b0 + b1
    denominator = E0 + E1 + T0 / CE * bt**2
    z_factor = E1 + T0 / CE * bt * b1
    columns = np.eye(2*n+1)
    h, z, bath = columns[:n], STRAIN_SCALE * columns[n:2*n], columns[-1]
    load = z_factor[:, None] * z + bt[:, None] * h
    common_stress = -np.sum(load / denominator[:, None], axis=0) / np.sum(1 / denominator)
    strain = (load + common_stress) / denominator[:, None]
    theta = h - T0 / CE * bt[:, None] * strain + T0 / CE * b1[:, None] * z
    temperature = np.vstack([theta, bath])
    generator = np.empty_like(columns)
    generator[:n] = -(heat @ temperature)[:n] / (CE * volume)
    generator[n:2*n] = (E1 * (strain-z) - b1[:, None] * theta) / eta[:, None] / STRAIN_SCALE
    generator[-1] = -(heat @ temperature)[-1] / CB
    return generator, temperature


def initial(n, prep, chi):
    x, b0, b1, *_ = geometry(n, 145., 0., chi)
    a, b, c, bath = prep
    theta = a + b * np.cos(np.pi*x/L) + c*np.cos(2*np.pi*x/L)
    strain = (b0*theta - np.mean(b0*theta)) / E0
    z = strain - b1*theta/E1
    h = theta + T0/CE * ((b0+b1)*strain-b1*z)
    return np.r_[theta, z/STRAIN_SCALE, bath], np.r_[h, z/STRAIN_SCALE, bath]


def evolve(generator, states, times=TIMES):
    values, vectors = eig(generator)
    weights = solve(vectors, states)
    evolved = np.stack([(vectors @ (np.exp(values*t)[:, None]*weights)).real for t in times])
    return evolved, float(np.max(values.real))


def readout(temperature):
    n = temperature.shape[1]-1
    x = (np.arange(n)+.5)*L/n
    weights = np.stack([np.ones(n)/n, np.cos(np.pi*x/L)/n, np.cos(2*np.pi*x/L)/n])
    rod = np.einsum('on,tnp->top', weights, temperature[:, :n])
    return np.concatenate([rod, temperature[:, -1:, :]], axis=1)


def run_case(n, k, contact, chi, reference=False):
    exact, info = forward(n, k, contact, chi)
    shortcut, _ = forward(n, k, contact, chi, True)
    states = np.column_stack([initial(n, p, chi)[0] for p in PREPS])
    true_path, growth_true = evolve(exact, states)
    wrong_path, growth_wrong = evolve(shortcut, states)
    selection = np.r_[np.arange(n), 2*n]
    actual = readout(true_path[:, selection])
    wrong = readout(wrong_path[:, selection])
    x, b0, b1, eta, heat, volume, st, sz = info
    theta, z = true_path[:, :n], STRAIN_SCALE * true_path[:, n:2*n]
    strain = np.einsum('ij,tjp->tip', st, theta) + np.einsum('ij,tjp->tip', sz, z)
    stress = (E0+E1)*strain-E1*z-(b0+b1)[None, :, None]*theta
    energy = volume*np.sum(CE*theta+T0*((b0+b1)[None, :, None]*strain-b1[None, :, None]*z), axis=1) + CB*true_path[:, -1]
    mechanical = E0*strain**2 + E1*(strain-z)**2
    availability = .5*volume*np.sum(mechanical+CE/T0*theta**2, axis=1)+CB/(2*T0)*true_path[:, -1]**2
    row = dict(n=n, conductivity=k, contact=contact, chi=chi,
               exact=actual.tolist(), shortcut=wrong.tolist(),
               rmse_kelvin=float(np.sqrt(np.mean((actual[1:]-wrong[1:])**2))),
               max_gap_kelvin=float(np.max(np.abs(actual-wrong))),
               exact_max_eigenvalue=growth_true, shortcut_max_eigenvalue=growth_wrong,
               max_stress_variation_pa=float(np.max(np.ptp(stress, axis=1))),
               max_mean_strain=float(np.max(np.abs(np.mean(strain, axis=1)))),
               energy_drift_joule=float(np.max(np.abs(energy-energy[0]))),
               max_availability_increase_joule=float(np.max(np.diff(availability, axis=0))))
    if reference:
        ref, output = entropy_reference(n, k, contact, chi)
        starts = np.column_stack([initial(n, p, chi)[1] for p in PREPS])
        path, _ = evolve(ref, starts)
        other = readout(np.einsum('ij,tjp->tip', output, path))
        row['entropy_reference_max_error_kelvin'] = float(np.max(np.abs(actual-other)))
    return row


def main():
    started = time.perf_counter()
    rows = [run_case(40, k, contact, chi, True)
            for chi in [0., .01, .1, .25, .5, 1.]
            for k in [80., 145., 220.] for contact in [0., .4, 1.1]]
    refined = []
    for chi, k, contact in [(0., 145., .4), (.25, 80., 0.), (.5, 145., .4), (1., 220., 1.1)]:
        coarse = next(r for r in rows if r['chi']==chi and r['conductivity']==k and r['contact']==contact)
        fine = run_case(80, k, contact, chi, True)
        fine['40_to_80_max_error_kelvin'] = float(np.max(np.abs(np.array(coarse['exact'])-np.array(fine['exact']))))
        refined.append(fine)
    report = dict(status='bounded_screen_not_task_validation', preps=PREPS, times=TIMES.tolist(), cases=rows, refinement=refined,
                  seconds=time.perf_counter()-started,
                  summary={key: max(r[key] for r in rows) for key in ['max_gap_kelvin', 'entropy_reference_max_error_kelvin', 'energy_drift_joule', 'max_stress_variation_pa', 'max_mean_strain', 'exact_max_eigenvalue', 'shortcut_max_eigenvalue', 'max_availability_increase_joule']})
    Path(__file__).with_name('report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'seconds':report['seconds'], 'cases':len(rows), 'summary':report['summary'],
                      'max_rmse_by_chi': {str(c):max(r['rmse_kelvin'] for r in rows if r['chi']==c) for c in [0.,.01,.1,.25,.5,1.]}}, indent=2))


if __name__ == '__main__':
    main()
