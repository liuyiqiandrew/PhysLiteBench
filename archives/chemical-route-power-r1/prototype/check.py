"""Author-only prototype: chemical regeneration power versus aggregate entropy.

No task or agent evaluation. Rates have units s^-1, k_B T=1, and all
chemical drops/state free energies are in the same energy unit.
"""
from itertools import product
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.linalg import eigvals


def physical_rates(energies, attempts, affinity, scale=1.0):
    energies = np.asarray(energies, dtype=float)
    attempts = np.asarray(attempts, dtype=float)
    drop = np.asarray(affinity)[None, :] - (np.roll(energies, -1) - energies)[:, None]
    return scale * attempts * np.exp(drop / 2), scale * attempts * np.exp(-drop / 2)


def models(energies, attempts, affinity, scale=1.0):
    forward, reverse = physical_rates(energies, attempts, affinity, scale)
    f, r = forward.sum(axis=1), reverse.sum(axis=1)
    generator = np.zeros((3, 3))
    for i in range(3):
        j = (i + 1) % 3
        generator[j, i] += f[i]
        generator[i, j] += r[i]
        generator[i, i] -= f[i]
        generator[j, j] -= r[i]
    lhs = generator.copy()
    lhs[-1] = 1
    p = np.linalg.solve(lhs, [0., 0., 1.])
    jp = p[:, None] * forward
    jm = np.roll(p, -1)[:, None] * reverse
    currents = jp - jm
    edge_forward, edge_reverse = jp.sum(axis=1), jm.sum(axis=1)
    coarse = np.sum((edge_forward-edge_reverse) * np.log(edge_forward/edge_reverse))
    physical = np.sum(currents * np.asarray(affinity)[None, :])
    resolved_entropy = np.sum(currents * np.log(jp/jm))
    return float(physical), float(coarse), p, generator, currents, float(resolved_entropy)


def counted_power(energies, attempts, affinity, scale=1.0, step=1e-5):
    """Independent channel-marked characteristic generator; no stationary solve.

The eigenvalue that continues zero is the cumulant rate. Its first derivative
counts actual signed fuel conversion, before any routes are summed.
"""
    tilted = np.zeros((3, 3), complex)
    for edge in range(3):
        target = (edge + 1) % 3
        for channel in range(2):
            chemical_drop = affinity[channel]
            forward_rate = scale * attempts[edge][channel] * np.exp(
                (chemical_drop + energies[edge] - energies[target]) / 2)
            reverse_rate = scale * attempts[edge][channel] * np.exp(
                (-chemical_drop - energies[edge] + energies[target]) / 2)
            tilted[target, edge] += forward_rate * np.exp(1j*step*chemical_drop)
            tilted[edge, target] += reverse_rate * np.exp(-1j*step*chemical_drop)
            tilted[edge, edge] -= forward_rate
            tilted[target, target] -= reverse_rate
    values = eigvals(tilted)
    pole = values[np.argmin(abs(values))]
    return float(pole.imag / step)


def calibration_inputs():
    out = []
    for a in (.4, .9, 1.5):
        for e in ((0., -.3, .2), (0., .3, -.2), (0., .1, .3)):
            for flip in (False, True):
                v = np.array([[.6, 1.2], [1.3, .7], [.8, 1.1]])
                if flip:
                    v = v * np.array([1.15, .8, 1.1])[:, None]
                out.append((np.array(e), v.copy(), np.array([a, a])))
    return out


def run():
    rng = np.random.default_rng(412605)
    report = {'status':'science_prototype_only', 'model_evaluations':0,
              'common_scale_interval':[.6, 1.4],
              'temperature':1.,
              'hidden_domain':{'energies':[0., [-.3,.3],[-.3,.3]],
                               'attempts_each':[.6,1.4],
                               'affinity_0':[1.6,2.4], 'affinity_1':[.2,.6]},
              'units':'energies in kBT=1; known attempts and common scale multiply to inverse seconds; measured regeneration power is energy/time'}
    largest_cal = 0.
    derivatives = []
    largest_recovery = 0.
    cal = calibration_inputs()
    clean = np.array([models(*e)[0] for e in cal])
    for scale in np.linspace(.6, 1.4, 41):
        physical = np.array([models(*e, scale)[0] for e in cal])
        coarse = np.array([models(*e, scale)[1] for e in cal])
        largest_cal = max(largest_cal, float(np.max(abs(physical-coarse))))
        fitted = clean @ physical / (clean @ clean)
        largest_recovery = max(largest_recovery, abs(fitted-scale))
    derivatives = clean.tolist()
    report['calibration']={'settings':len(cal),'nonzero_affinities':[.4,.9,1.5],
                           'both_routes_active':True,'max_model_difference':largest_cal,
                           'min_d_power_d_scale':min(derivatives),'max_d_power_d_scale':max(derivatives),
                           'noiseless_scales_checked':41,'max_scale_recovery_error':largest_recovery,
                           'global_identifiability':'P(scale)=scale*P(1), with every calibrated P(1)>0; weighted objective is strictly convex',
                           'fixed_sigma_proposal':.003,
                           'one_copy_fisher_information':float(clean@clean/.003**2)}
    min_gap = float('inf')
    max_gap = 0.
    min_signal = float('inf')
    min_coarse = float('inf')
    min_population = 1.
    min_route_current = float('inf')
    max_balance = max_entropy_error = 0.
    worst = None
    checks = []
    for e1,e2,a1,a2,*vv in product(*([[-.3,.3]]*2+[[1.6,2.4],[.2,.6]]+[[.6,1.4]]*6)):
        e = np.array([0., e1, e2]); v = np.array(vv).reshape(3,2); a = np.array([a1,a2])
        physical, coarse, p, gen, currents, entropy = models(e,v,a)
        gap = (physical-coarse)/physical
        if gap < min_gap:
            min_gap = gap
            worst = {'energies':e.tolist(),'attempts':v.tolist(),'affinities':a.tolist(),
                     'physical':physical,'coarse':coarse,'relative_gap':gap}
        max_gap=max(max_gap,gap); min_signal=min(min_signal,physical); min_coarse=min(min_coarse,coarse)
        min_population=min(min_population,p.min()); min_route_current=min(min_route_current,currents.min())
        max_balance=max(max_balance,np.max(abs(gen@p)),abs(np.sum(currents*(np.roll(e,-1)-e)[:,None])))
        max_entropy_error=max(max_entropy_error,abs(physical-entropy))
        checks.append((e,v,a))
    report['corners']={'count':len(checks),'min_relative_gap':min_gap,'max_relative_gap':max_gap,
                       'min_physical_power_at_scale1':min_signal,'min_coarse_power_at_scale1':min_coarse,
                       'min_stationary_probability':float(min_population),
                       'min_individual_route_current':float(min_route_current),
                       'max_stationary_energy_probability_residual':float(max_balance),
                       'max_chemical_power_minus_resolved_entropy':float(max_entropy_error),
                       'least_gap_case':worst,
                       'interpretation':'Both fuel drops are positive. A minority route may run backward locally; net physical and aggregate powers stay well away from zero.'}
    random_cases=[]
    for _ in range(128):
        random_cases.append((np.r_[0.,rng.uniform(-.3,.3,2)],rng.uniform(.6,1.4,(3,2)),
                            np.array([rng.uniform(1.6,2.4),rng.uniform(.2,.6)])))
    max_ref=max_refinement=0.
    min_random_gap=1.
    for e,v,a in random_cases+cal:
        scale=rng.uniform(.6,1.4)
        exact,coarse,*_=models(e,v,a,scale)
        r1=counted_power(e,v,a,scale,2e-5); r2=counted_power(e,v,a,scale,1e-5)
        ref=(4*r2-r1)/3
        max_ref=max(max_ref,abs(ref-exact)/max(1.,abs(exact)))
        max_refinement=max(max_refinement,abs(r2-r1)/max(1.,abs(exact)))
        if a[0]!=a[1]: min_random_gap=min(min_random_gap,(exact-coarse)/exact)
    report['independent_channel_counting']={'cases':len(random_cases)+len(cal),
                                           'method':'dominant eigenvalue of route-marked characteristic generator, two counting steps and Richardson derivative',
                                           'max_scaled_error':max_ref,'max_counting_step_change':max_refinement,
                                           'min_random_relative_gap':min_random_gap}
    max_shift=max_exchange=max_scale=max_equilibrium=0.
    for e,v,a in random_cases[:32]:
        exact,coarse,*_=models(e,v,a)
        shift=models(e+7.,v,a)
        swap=models(e,v[:,::-1],a[::-1])
        scaled=models(e,v,a,1.3)
        equil=models(e,v,np.zeros(2))
        max_shift=max(max_shift,abs(exact-shift[0]),abs(coarse-shift[1]))
        max_exchange=max(max_exchange,abs(exact-swap[0]),abs(coarse-swap[1]))
        max_scale=max(max_scale,abs(scaled[0]-1.3*exact),abs(scaled[1]-1.3*coarse))
        max_equilibrium=max(max_equilibrium,abs(equil[0]),abs(equil[1]))
    report['limits']={'cases':32,'energy_origin_invariance':max_shift,
                      'channel_exchange_invariance':max_exchange,'common_rate_scaling':max_scale,
                      'equilibrium_zero_power':max_equilibrium}
    # Uniform ring examples require neither a stationary matrix solve nor eigenvalues.
    examples=[]
    for f,r in (([4.,2.],[1.,1.]),([6.,1.],[1.,4.])):
        f=np.array(f);r=np.array(r);a=np.log(f/r);v=np.tile(np.sqrt(f*r),(3,1))
        measured,source,*_=models(np.zeros(3),v,a)
        exact=float((f-r)@a);approx=float((sum(f)-sum(r))*np.log(sum(f)/sum(r)))
        examples.append({'forward':f.tolist(),'reverse':r.tolist(),'physical':measured,'coarse':source,
                         'relative_gap':(measured-source)/measured,
                         'analytic_error':max(abs(exact-measured),abs(approx-source))})
    report['uniform_ring_examples']=examples
    assert largest_cal<1e-12 and largest_recovery<1e-12
    assert min_gap>.04 and min_signal>.1 and min_coarse>.1
    assert min_population>0 and max_balance<1e-12 and max_entropy_error<1e-12
    assert max_ref<1e-7 and min_random_gap>.04
    report['all_checks_passed']=True
    report['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    Path(__file__).with_name('report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    run()
