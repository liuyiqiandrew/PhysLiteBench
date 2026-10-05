"""Supplement the preserved first screen without changing it."""
import json
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from check import (AREA, CB, CE, E0, E1, PREPS, STRAIN_SCALE, T0, TIMES,
                   entropy_reference, evolve, forward, initial, readout, run_case)


def calibration(k, n=40):
    generator, _ = forward(n, k, 0., 0.)
    states = initial(n, PREPS[0], 0.)[0][:, None]
    path, _ = evolve(generator, states, [.2, 1., 4., 12., 40.])
    return readout(path[:, np.r_[np.arange(n), 2*n]])[:, 1, 0]


def diagnostics(n, k, contact, chi):
    true, info = forward(n, k, contact, chi)
    source, _ = forward(n, k, contact, chi, True)
    x, b0, b1, eta, heat, volume, st, sz = info
    starts = np.column_stack([initial(n, p, chi)[0] for p in PREPS])
    late, _ = evolve(true, starts, [1e5])
    late_source, _ = evolve(source, starts, [1e5])
    path, _ = evolve(source, starts)
    theta, z = path[:, :n], STRAIN_SCALE*path[:, n:2*n]
    strain = np.einsum('ij,tjp->tip', st, theta)+np.einsum('ij,tjp->tip', sz, z)
    s_rel_T0 = (CE+T0*b1**2/E1)[None, :, None]*theta+T0*b0[None, :, None]*strain
    own_energy = volume*np.sum(s_rel_T0, axis=1)+CB*path[:, -1]
    physical_energy = volume*np.sum(CE*theta+T0*((b0+b1)[None,:,None]*strain-b1[None,:,None]*z), axis=1)+CB*path[:, -1]
    arm_stress = E1*(strain-z)-b1[None, :, None]*theta
    entropy_difference_T0 = T0*b1[None, :, None]*arm_stress/E1
    physical_s_T0 = CE*theta+T0*((b0+b1)[None,:,None]*strain-b1[None,:,None]*z)
    z_rate_initial = true[n:2*n] @ starts * STRAIN_SCALE
    capacities = []
    for c in [0., .01, .25, .5, 1.]:
        bt = (1+c)*b0
        a = (np.eye(n)-np.ones((n,n))/n)*bt[None,:]/(E0+E1)
        capacities.append(float(np.linalg.eigvalsh(np.diag(CE+T0*(c*b0)**2/E1)+T0*b0[:,None]*a)[0]))
    return dict(n=n, conductivity=k, contact=contact, chi=chi,
                initial_max_z_rate=float(np.max(np.abs(z_rate_initial))),
                final_temperature_agreement_kelvin=float(np.max(np.abs(late[:,np.r_[np.arange(n),2*n]]-late_source[:,np.r_[np.arange(n),2*n]]))),
                own_linear_energy_drift_joule=float(np.max(np.abs(own_energy-own_energy[0]))),
                physical_linear_energy_drift_joule=float(np.max(np.abs(physical_energy-physical_energy[0]))),
                missing_entropy_identity_error=float(np.max(np.abs(physical_s_T0-s_rel_T0-entropy_difference_T0))),
                source_capacity_min_eigenvalue_over_chi=capacities)


def main():
    started = time.perf_counter()
    groups = []
    for k in np.linspace(80, 220, 15):
        for contact, prep in [(0.,1),(.4,2),(1.1,2)]:
            row = run_case(64, float(k), contact, 1.)
            exact, source = np.asarray(row['exact'])[2:,0,prep], np.asarray(row['shortcut'])[2:,0,prep]
            groups.append(dict(conductivity=float(k),contact=contact,prep=PREPS[prep],times=TIMES[2:].tolist(),exact=exact.tolist(),source=source.tolist(),rmse_kelvin=float(np.sqrt(np.mean((exact-source)**2))),rms_signal_kelvin=float(np.sqrt(np.mean(exact**2)))))
    ks = np.linspace(80, 220, 81)
    profiles = np.array([calibration(k) for k in ks])
    fits = []
    for k in np.linspace(80,220,9):
        truth = calibration(k)
        objective = np.sum((profiles-truth)**2, axis=1)
        fitted = minimize_scalar(lambda value:np.sum((calibration(value)-truth)**2),bounds=(80,220),method='bounded',options={'xatol':1e-9})
        fits.append(dict(true=float(k),fitted=float(fitted.x),relative_error=float(abs(fitted.x/k-1)),sampled_local_minima=int(np.sum((objective[1:-1]<objective[:-2])&(objective[1:-1]<objective[2:])))))
    refinement = []
    for k,contact in [(80.,0.),(145.,.4),(220.,1.1)]:
        lower, middle, higher = [run_case(n,k,contact,1.,n==160) for n in [40,80,160]]
        refinement.append(dict(conductivity=k,contact=contact,
                               error_40_to_80=float(np.max(np.abs(np.array(lower['exact'])-middle['exact']))),
                               error_80_to_160=float(np.max(np.abs(np.array(middle['exact'])-higher['exact']))),
                               entropy_reference_error=higher['entropy_reference_max_error_kelvin']))
    n,k,contact,chi=24,137.,.63,.73
    gen, output = entropy_reference(n,k,contact,chi)
    start = initial(n,PREPS[2],chi)[1]
    integrated = solve_ivp(lambda t,y:gen@y,(0.,80.),start,method='BDF',jac=gen,t_eval=TIMES,rtol=2e-10,atol=2e-12)
    first,_=forward(n,k,contact,chi)
    path,_=evolve(first,initial(n,PREPS[2],chi)[0][:,None])
    direct=readout(path[:,np.r_[np.arange(n),2*n]])[:,:,0]
    alternative=readout((output@integrated.y).T[:,:,None])[:,:,0]
    result=dict(status='bounded_followup_not_harness',groups=groups,
                minimum_group_rmse_kelvin=min(r['rmse_kelvin'] for r in groups),
                minimum_group_rms_signal_kelvin=min(r['rms_signal_kelvin'] for r in groups),
                calibration=dict(conductivities=ks.tolist(),predictions=profiles.tolist(),
                                 maximum_sampled_derivative=float(np.max(np.diff(profiles,axis=0)/np.diff(ks)[:,None])),
                                 minimum_absolute_sampled_derivative=float(np.min(np.abs(np.diff(profiles,axis=0)/np.diff(ks)[:,None]))),
                                 fits=fits,qualification='Sampled monotonicity and recovery only; no global analytical injectivity proof or noise validation.'),
                refinements=refinement,
                closure_diagnostics=[diagnostics(48,145.,contact,chi) for contact in [0.,.4] for chi in [0.,.5,1.]],
                independent_time_integration=dict(success=integrated.success,max_error_kelvin=float(np.max(np.abs(direct-alternative)))),seconds=time.perf_counter()-started)
    Path(__file__).with_name('followup-report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['groups','calibration','closure_diagnostics']},indent=2))
    print('calibration recovery',max(r['relative_error'] for r in fits))


if __name__=='__main__':
    main()
