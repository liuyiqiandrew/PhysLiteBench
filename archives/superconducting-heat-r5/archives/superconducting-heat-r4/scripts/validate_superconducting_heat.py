"""Validate stationary spin populations against independent matrix transitions."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/superconducting-heat'


def module(path):
    spec = importlib.util.spec_from_file_location('validation_'+str(abs(hash(str(path)))), path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def local_control(source):
    with tempfile.TemporaryDirectory(prefix='physlite-gated-control-') as temporary:
        app, tests = Path(temporary)/'app', Path(temporary)/'tests'
        shutil.copytree(TASK/'environment', app)
        shutil.copytree(TASK/'tests', tests)
        shutil.copyfile(source, app/'model.py')
        completed = subprocess.run([sys.executable, '-m', 'pytest', '-q', str(app/'test_public.py'),
                                    str(tests/'test_hidden.py')], cwd=app, text=True,
                                   capture_output=True, env=dict(__import__('os').environ, PYTHONPATH=str(app)))
        return dict(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)


def arguments(e):
    return (e['left_temperature'], e['right_temperature'], e['phase'],
            e.get('amplitude', 0.), e.get('drive_energy', .3))


def spin_basis_checks(ref):
    """Rotate amplitudes, occupations and the measured spin operator together."""
    sigma = (np.array([[0.,1.],[1.,0.]],complex), np.array([[0.,-1j],[1j,0.]],complex), np.diag([1.,-1.]))
    axis = np.array([1.,2.,3.])/np.sqrt(14.)
    generator = sum(a*b for a,b in zip(axis,sigma))
    rotations = [np.cos(t/2)*np.eye(2)-1j*np.sin(t/2)*generator for t in (.4,1.7,2.8)]
    rl, rr = np.diag([.3,.08]), np.diag([.7,.16])
    z, identity = sigma[2], np.eye(2)
    def rates(matrix, left, right, axis, pair):
        if pair:
            l = (identity-left)@matrix@(identity-right).T@matrix.conj().T-left@matrix@right.T@matrix.conj().T
            r = (identity-right)@matrix.T@(identity-left).T@matrix.conj()-right@matrix.T@left.T@matrix.conj()
            return np.array([np.trace(l).real,np.trace(axis@l).real,np.trace(axis@r).real])
        l = left@matrix@matrix.conj().T-matrix@right@matrix.conj().T
        r = matrix.conj().T@left@matrix-right@matrix.conj().T@matrix
        return np.array([np.trace(l).real,-np.trace(axis@l).real,np.trace(axis@r).real])
    error = 0.
    for harmonic in (-1,0,1):
        t = ref.spin_matrices(.8)[harmonic]
        h = ref.SINGLET@ref.spin_matrices(.8)[-harmonic].conj()@ref.SINGLET.conj().T
        for pair in (False,True):
            a = (.8*.6*t+.6*.8*np.exp(.7j)*h)@ref.SINGLET if pair else .8*.8*t-.6*.6*np.exp(.7j)*h
            original = rates(a,rl,rr,z,pair)
            for u in rotations:
                changed = u.conj().T@a@(u.conj() if pair else u)
                actual = rates(changed,u.conj().T@rl@u,u.conj().T@rr@u,u.conj().T@z@u,pair)
                error = max(error,float(max(abs(actual-original))))
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--noise-trials', type=int, default=256)
    args = parser.parse_args()
    start = time.time()
    good, bad, ref = [module(p) for p in (TASK/'solution/model.py', ROOT/'scripts/superconducting_heat_baseline.py', TASK/'tests/reference.py')]
    metadata = json.loads((TASK/'tests/metadata.json').read_text())
    true, sigma, limit = ref.TRUE_PARAMETER, metadata['measurement_sigma'], metadata['prediction_limit']
    inputs = ref.calibration_inputs()
    exact = ref.predict(inputs, true)
    if args.generate:
        values = exact+np.random.default_rng(metadata['calibration_seed']).normal(0, sigma, len(inputs))
        records = [dict(input=e, value=float(y), sigma=sigma) for e,y in zip(inputs,values)]
        for directory in ('environment','tests'):
            (TASK/directory/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
    records = json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records == json.loads((TASK/'tests/data/calibration.json').read_text())
    assert [r['input'] for r in records] == inputs and all(r['sigma'] == sigma for r in records)
    old_calibration = ROOT/'archives/superconducting-heat-r3/tasks/superconducting-heat/environment/data/calibration.json'
    assert (TASK/'environment/data/calibration.json').read_bytes() == old_calibration.read_bytes()
    hidden = ref.hidden_inputs()
    truth = {name:ref.predict(es,true) for name,es in hidden.items()}
    def chi(model,rs):
        residual = (model.predict(inputs)-np.array([r['value'] for r in rs]))/sigma
        return float(residual@residual/(len(rs)-1))
    def scores(model):
        return {name:float(np.sqrt(np.mean((model.predict(es)-truth[name])**2)/np.mean(truth[name]**2))) for name,es in hidden.items()}
    report = dict(revision=4, calibration_preserved=True, noise_trials=args.noise_trials,
                  calibration_seed=metadata['calibration_seed'], noise_seed=metadata['noise_seed'], controls={})
    for label,source in (('oracle',good),('shortcut',bad)):
        model = source.Model().fit(records)
        result = dict(parameter=model.conductance, parameter_relative_error=abs(model.conductance/true-1),
                      calibration_chi2=chi(model,records), hidden=scores(model))
        assert result['calibration_chi2'] < 1.5 and result['parameter_relative_error'] < .03
        assert max(result['hidden'].values()) < limit if label == 'oracle' else min(result['hidden'].values()) > limit
        report['controls'][label] = result
    rng = np.random.default_rng(metadata['noise_seed'])
    parameters,chi2s,oracle_errors,source_errors = [],[],[],[]
    for _ in range(args.noise_trials):
        sample = [dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,exact+rng.normal(0,sigma,len(inputs)))]
        a,b = good.Model().fit(sample),bad.Model().fit(sample)
        assert abs(a.conductance-b.conductance) < 1e-10
        parameters.append(a.conductance);chi2s.append(chi(a,sample))
        oracle_errors.extend(scores(a).values());source_errors.extend(scores(b).values())
    assert max(chi2s) < 1.5 and max(abs(np.array(parameters)/true-1)) < .03
    assert max(oracle_errors) < limit and min(source_errors) > limit
    report['noise'] = dict(parameter_min=min(parameters),parameter_max=max(parameters),
        parameter_relative_error_max=float(max(abs(np.array(parameters)/true-1))), calibration_chi2_max=max(chi2s),
        oracle_prediction_error_max=max(oracle_errors),shortcut_prediction_error_min=min(source_errors),
        oracle_passes=args.noise_trials,shortcut_intended_failures=args.noise_trials)
    corners = [ref.experiment(l,r,p,a,w) for l in (.25,1.05) for r in (.25,1.05)
               for p in (0.,np.pi,-.7) for a in (0.,.4,1.) for w in (.1,.5,7.,12.)]
    rng = np.random.default_rng(63021)
    interiors = [ref.experiment(rng.uniform(.25,1.05),rng.uniform(.25,1.05),rng.uniform(-np.pi,np.pi),
                    rng.uniform(0,1),rng.uniform(*((.1,.5) if i%2 else (7,12)))) for i in range(32)]
    cases = sum(hidden.values(),[])+corners+interiors
    agreement,mu_agreement,reference_refinement,ref_spin_error = [],[],[],[]
    energy,spinwork,entropy,entropy_identity,balance,dominance,mus,ratios,expansions = [],[],[],[],[],[],[],[],[]
    passivity,phase_reversal,chirality_heat = [],[],[]
    original = []
    for e in cases:
        a = arguments(e)
        value = good.stationary_state(*a,true)
        other = ref.energy_components(*a,conductance=true)
        original.append(value['heat'][0])
        agreement.append(abs(value['heat'][0]-other['heat']))
        mu_agreement.append(float(max(abs(value['polarization']-other['polarization']))))
        ref_spin_error.append(float(max(abs(value['spin']-other['spin'])))/(good.RATE_CONVERSION*true))
        energy.append(float(abs(sum(value['heat']))))
        spinwork.append(abs(value['heat'][2]-ref.KB*a[4]/2*sum(value['spin'])*1e12))
        entropy.append(value['tunneling_entropy']+value['relaxation_entropy'])
        entropy_identity.append(abs(entropy[-1]+value['heat'][0]*1e-12/a[0]+value['heat'][1]*1e-12/a[1]))
        balance.append(float(max(abs(value['residual'])))/(good.RATE_CONVERSION*true))
        j = value['jacobian']
        dominance.append(float(min(-j[0,0]-abs(j[1,0]),-j[1,1]-abs(j[0,1]))))
        mus.append(float(max(abs(value['polarization']))));ratios.append(mus[-1]/a[4]);expansions.append(value['search_expansions'])
        reverse = good.stationary_state(a[0],a[1],-a[2],a[3],a[4],true)
        phase_reversal.append(abs(value['heat'][0]-reverse['heat'][0]))
        if a[0] == a[1]:
            passivity.append(value['heat'][2])
    refinement = []
    for e,y in zip(cases,original):
        refined = good.stationary_state(*arguments(e),true,nodes=288)
        refinement.append(abs(refined['heat'][0]-y))
    for e in interiors[:8]:
        a = arguments(e)
        v = ref.energy_components(*a,conductance=true)
        t = ref.energy_components(*a,conductance=true,tolerance=1e-12)
        reference_refinement.append(abs(v['heat']-t['heat']))
        chiral = ref.energy_components(*a,conductance=true,chirality=-1)
        chirality_heat.append(abs(v['heat']-chiral['heat']))
        assert max(abs(v['polarization']+chiral['polarization'])) < 1e-8
    extra_conductance = []
    multiple_start = []
    for e in interiors[:12]:
        a = arguments(e)
        base = good.stationary_state(*a,true)
        for initial in ((a[4]/3,a[4]/3),(-a[4]/3,-a[4]/3),(a[4]/2,-a[4]/2)):
            other = good.stationary_state(*a,true,initial=initial)
            multiple_start.append(float(max(abs(base['polarization']-other['polarization']))))
        for conductance in (20.,80.):
            expected = ref.energy_components(*a,conductance=conductance)
            actual = good.stationary_state(*a,conductance)
            extra_conductance.append(abs(expected['heat']-actual['heat'][0]))
    normal = []
    for tl,tr,a,w in ((.25,.25,1.,8.),(.4,.8,.7,7.),(.8,.4,.3,10.),(.25,1.05,1.,.3)):
        c,r = good.RATE_CONVERSION*true,good.QUASIPARTICLE_DENSITY/good.NORMAL_POLARIZATION_TIME
        mu = c*a*a*w/(2*(r+c*a*a))
        thermal = lambda temperature:np.pi**2*temperature**2/6+mu**2/2
        q = true*1e6*(ref.KB/ref.CHARGE)**2*((1+a*a)*(thermal(tl)-thermal(tr))-a*a*w*w/2+a*a*w*mu)
        work = true*1e6*(ref.KB/ref.CHARGE)**2*a*a*w*(w-2*mu)
        exact_state = ref.energy_components(tl,tr,.7,a,w,conductance=true,normal=True)
        numerical = good.stationary_state(tl,tr,.7,a,w,true,normal=True)
        normal.append(dict(mu_closed_K=mu,mu_oracle_error_K=float(max(abs(numerical['polarization']-mu))),
          mu_reference_error_K=float(max(abs(exact_state['polarization']-mu))),heat_oracle_error_pW=abs(numerical['heat'][0]-q),
          heat_reference_error_pW=abs(exact_state['heat']-q),work_error_pW=abs(numerical['heat'][2]-work)))
    matrix_errors,probabilities = [],[]
    for dl,dr,el,er in ((2.,3.,2.1,3.2),(1.3,2.8,4.,5.),(0.,0.,1.,3.)):
        for phase in (0.,.7,np.pi/2,np.pi):
            for n in (-1,0,1):
                for creation in (False,True):
                    value = ref.branch_probability(el,er,dl,dr,phase,n,.8,creation=creation)
                    weight = 1. if n == 0 else .8**2/2
                    sign = (-1 if n == 0 else 1)*(-1 if creation else 1)
                    expected = np.zeros((2,2))
                    coherence = weight*(1+sign*dl*dr*np.cos(phase)/(el*er))
                    if n == 0:
                        for spin in (0,1):
                            expected[spin,1-spin if creation else spin] = coherence/2
                    else:
                        i = 0 if n == 1 else 1
                        expected[i,i if creation else 1-i] = coherence
                    matrix_errors.append(float(max(abs(value-expected).ravel())));probabilities.extend(value.ravel())
    source_reference = []
    for e in sum(hidden.values(),[]):
        expected = ref.energy_components(*arguments(e),conductance=true,force_unpolarized=True)['heat']
        source_reference.append(abs(bad.predict_at([e],true)[0]-expected))
    static = float(max(abs(good.predict_at(inputs,true)-bad.predict_at(inputs,true))))
    fast = {}
    for tau in (1e-12,1e-14):
        errors,polarizations = [],[]
        for e in sum(hidden.values(),[]):
            v = good.stationary_state(*arguments(e),true,relaxation_time=tau)
            errors.append(abs(v['heat'][0]-bad.predict_at([e],true)[0]));polarizations.extend(abs(v['polarization']))
        fast[str(tau)] = dict(maximum_heat_error_pW=max(errors),maximum_polarization_K=float(max(polarizations)))
    spinflip = []
    from scipy.integrate import quad
    for temperature,delta,mu in ((.25,2.,1.),(.8,3.,2.),(.9,1.,-.3)):
        def integrand(z):
            e = delta+z*z
            dos = e/np.sqrt((e-delta)*(e+delta))
            rate = np.sqrt((e-delta)*(e+delta))/e/good.NORMAL_POLARIZATION_TIME
            return 2*z*dos*rate*(good.expit((mu-e)/temperature)-good.expit((-mu-e)/temperature))
        actual = quad(integrand,0.,np.sqrt(80*temperature+abs(mu)),epsabs=1e-6,epsrel=1e-12)[0]*good.QUASIPARTICLE_DENSITY
        expected = good.QUASIPARTICLE_DENSITY/good.NORMAL_POLARIZATION_TIME*good.spin_loss(temperature,delta,mu)[0]
        spinflip.append(float(abs(actual-expected)/abs(expected)))
    basis = spin_basis_checks(ref)
    assert max(agreement+reference_refinement+extra_conductance+refinement) < 1e-8
    assert max(mu_agreement+multiple_start) < 1e-8 and max(ref_spin_error) < 1e-8
    assert max(energy+spinwork+phase_reversal+chirality_heat) < 1e-10
    assert max(balance) < 1e-10 and min(dominance) > 0
    assert min(entropy) >= -1e-24 and min(passivity) >= -1e-12 and max(entropy_identity) < 1e-22
    assert max(matrix_errors) < 1e-13 and min(probabilities) >= 0 and basis < 1e-13
    assert max(source_reference) < 1e-8 and static == 0.
    assert max(spinflip) < 1e-12
    assert fast['1e-14']['maximum_heat_error_pW'] < 5e-8 and fast['1e-14']['maximum_heat_error_pW'] < fast['1e-12']['maximum_heat_error_pW']/90
    assert max(max(v[k] for k in v if k.endswith('_error_K') or k.endswith('_error_pW')) for v in normal) < 1e-8
    report['physics'] = dict(test_settings=len(cases),oracle_reference_max_pW=max(agreement),
        oracle192_vs288_nodes_max_pW=max(refinement),reference_tolerance_refinement_max_pW=max(reference_refinement),
        conductance_boundary_reference_max_pW=max(extra_conductance),polarization_reference_max_K=max(mu_agreement),
        multiple_initial_state_max_K=max(multiple_start),stationarity_normalized_max=max(balance),
        spin_reference_normalized_max=max(ref_spin_error),strict_column_dominance_min=min(dominance),
        maximum_polarization_K=max(mus),maximum_abs_polarization_over_drive=max(ratios),search_expansions_max=max(expansions),
        energy_work_residual_max_pW=max(energy),spin_work_identity_max_pW=max(spinwork),
        total_entropy_min_W_per_K=min(entropy),entropy_heat_identity_max_W_per_K=max(entropy_identity),
        equal_temperature_gate_work_min_pW=min(passivity),phase_reversal_max_pW=max(phase_reversal),
        chirality_heat_max_pW=max(chirality_heat),normal_state_checks=normal,
        spin_matrix_probability_max=max(matrix_errors),spin_basis_covariance_max=basis,minimum_transition_probability=min(probabilities),
        unpolarized_shortcut_matrix_reference_max_pW=max(source_reference),calibration_equivalence_max_pW=static,
        microscopic_spinflip_relative_max=max(spinflip),fast_relaxation_limit=fast,
        drive_domains_K=[[.1,.5],[7.,12.]],gap_sum_bounds_K=[4.292092116356208,5.462734949908928],
        gap_difference_bounds_K=[.9591028979493732,2.1297457315020933])
    local = {'oracle':local_control(TASK/'solution/model.py'),'shortcut':local_control(ROOT/'scripts/superconducting_heat_baseline.py')}
    assert local['oracle']['returncode'] == 0
    assert local['shortcut']['returncode'] == 1 and '3 failed, 4 passed' in local['shortcut']['stdout']
    report['local_controls'] = local
    report['seconds'] = time.time()-start
    files = list(TASK.rglob('*'))+[ROOT/'scripts/superconducting_heat_baseline.py',Path(__file__)]
    report['source_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in files if p.is_file() and '__pycache__' not in str(p)}
    output = ROOT/'results/superconducting-heat-r4-validation.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS',output,report['seconds'])
    print(json.dumps(report['controls'],indent=2))
    print(json.dumps(report['physics'],indent=2))


if __name__ == '__main__':
    main()
