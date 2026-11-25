"""Read-only numerical/causal review of two completed native r4 submissions.

No model calls. Diagnostic variants are executed from strings in memory;
frozen source and trial artifacts are never edited.
"""
import sys
sys.dont_write_bytecode = True
import hashlib, importlib.util, json, pathlib, time, types
import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT = pathlib.Path('/private/tmp/physlitebench-hardening-20261005')
STAGE = ROOT/'staging/luna-high-hardening-20261005'
JOB = ROOT/'jobs/superconducting-heat-r4-plain-20261006-030806'
OUT = STAGE/'results/superconducting-heat-r4-terminal-review-numerics.json'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def memory_module(source, name):
    obj = types.ModuleType(name)
    exec(compile(source, '<diagnostic '+name+'>', 'exec'), obj.__dict__)
    return obj

def rms(actual, truth):
    return float(np.sqrt(np.mean((actual-truth)**2)/np.mean(truth**2)))

def args(e):
    return (e['left_temperature'], e['right_temperature'], e['phase'],
            e.get('amplitude', 0.), e.get('drive_energy', .3))

def ref_value(e, conductance, tol=1e-10):
    return ref.energy_components(*args(e), conductance=conductance,
                                 tolerance=tol)

def oracle_value(e, conductance):
    if e.get('amplitude', 0.):
        return oracle.stationary_state(*args(e), conductance)['heat'][0]
    return oracle.predict_at([e], conductance)[0]

def numerical_summary(actual, truth):
    delta = np.abs(np.asarray(actual)-np.asarray(truth))
    scaled = delta/np.maximum(np.abs(truth), .01)
    return dict(max_absolute_pW=float(delta.max()),
                rms_relative=float(np.sqrt(np.mean(delta**2)/np.mean(np.asarray(truth)**2))),
                max_error_scaled_by_max_truth_or_0_01_pW=float(scaled.max()))

start = time.perf_counter()
passpath = JOB/'superconducting-heat__eiHfe6V/artifacts/app/model.py'
failpath = JOB/'superconducting-heat__gPoQqPc/artifacts/app/model.py'
source_hashes = {str(p.relative_to(ROOT)):digest(p) for p in (passpath,failpath)}
passer = module(passpath, 'super_r4_passer')
failing = module(failpath, 'super_r4_failure')
oracle = module(JOB/'frozen-task/solution/model.py', 'super_r4_oracle')
ref = module(JOB/'frozen-task/tests/reference.py', 'super_r4_reference')
records = json.loads((JOB/'frozen-task/environment/data/calibration.json').read_text())
fitted = {name:m.Model().fit(records).conductance for name,m in [('pass',passer),('failure',failing)]}
assert abs(fitted['pass']-fitted['failure']) < 1e-12
g = fitted['pass']
source = failpath.read_text()
weight_old = 'a2 = float(amplitude) ** 2 / 2.0'
pair_old = 'coherence = (energy * other - dl * dr * np.cos(phase)) / denominator'
assert source.count(weight_old) == 2
assert source.count(pair_old) == 1
weight_source = source.replace(weight_old, 'a2 = float(amplitude) ** 2')
pair_source = source.replace(pair_old, 'coherence = (energy * other - dl * dr * np.cos(phase))')
both_source = weight_source.replace(pair_old, 'coherence = (energy * other - dl * dr * np.cos(phase))')
variants = dict(original=failing,
                driven_normalization_only=memory_module(weight_source,'normalization_only'),
                pair_edge_denominator_only=memory_module(pair_source,'pair_edge_only'),
                both_algebra_repairs=memory_module(both_source,'both_algebra_repairs'))
report = dict(fitted_conductances=fitted, source_hashes=source_hashes,
              diagnostic_source_hashes={name:hashlib.sha256(s.encode()).hexdigest()
                                        for name,s in [('driven_normalization_only',weight_source),
                                                       ('pair_edge_denominator_only',pair_source),
                                                       ('both_algebra_repairs',both_source)]},
              changes='Two a2 coefficients doubled within the half-size per-spin rate/power convention; remove the duplicate regular pair denominator. No root-solver, fitted parameter, coherence sign, spin channel, or quadrature changes.',
              hidden_same_parameter={}, hidden_against_true_parameter={},
              offgrid_oracle=[], independent_reference=[])

for group, experiments in ref.hidden_inputs().items():
    exact = np.array([ref_value(e,g)['heat'] for e in experiments])
    true = np.array([ref_value(e,45.)['heat'] for e in experiments])
    values = {name:m.predict_at(experiments,g) for name,m in variants.items()}
    values['passer'] = passer.predict_at(experiments,g)
    report['hidden_same_parameter'][group] = {name:rms(v,exact) for name,v in values.items()}
    report['hidden_against_true_parameter'][group] = {name:rms(v,true) for name,v in values.items()}
    print('HIDDEN',group,report['hidden_same_parameter'][group],flush=True)

corners = [ref.experiment(l,r,p,1.,w)
           for l in (.25,1.05) for r in (.25,1.05)
           for p in (-np.pi,0.,np.pi/2) for w in (.1,.5,7.,12.)]
rng = np.random.default_rng(193411)
interiors = [ref.experiment(rng.uniform(.25,1.05),rng.uniform(.25,1.05),
                           rng.uniform(-np.pi,np.pi),rng.uniform(.025,1.),
                           rng.uniform(.1,.5) if i%2==0 else rng.uniform(7.,12.))
             for i in range(32)]
extra = [ref.experiment(.25,1.05,.3,0.,.5),ref.experiment(1.05,.25,-.4,0.,12.),
         ref.experiment(.31,.98,.8,.001,.37),ref.experiment(.98,.31,-1.2,.001,9.1)]
all_cases = corners+interiors+extra
screen = [(e,g) for e in all_cases]+[(e,k) for e in corners[::6]+interiors[::8] for k in (20.,80.)]
for i,(e,k) in enumerate(screen):
    expected = oracle_value(e,k)
    actual = passer.predict_at([e],k)[0]
    repaired = variants['both_algebra_repairs'].predict_at([e],k)[0]
    report['offgrid_oracle'].append(dict(input=e,conductance=k,expected=expected,
                                       passer=float(actual),repaired_failure=float(repaired)))
report['offgrid_oracle_summary'] = {
    name:numerical_summary([r[name] for r in report['offgrid_oracle']],
                           [r['expected'] for r in report['offgrid_oracle']])
    for name in ('passer','repaired_failure')}
print('SCREEN',len(screen),report['offgrid_oracle_summary'],flush=True)

# The explicit 2x2 transition reference independently checks both frequency
# bands, extreme temperatures/phases and G-dependent stationary populations.
checks = [(e,g) for e in corners[::4]+interiors[::4]+extra]
checks += [(e,k) for e in (corners[0],corners[-1],interiors[7]) for k in (20.,80.)]
for i,(e,k) in enumerate(checks):
    exact = ref_value(e,k)
    p = passer.predict_at([e],k)[0]
    repaired = variants['both_algebra_repairs'].predict_at([e],k)[0]
    fail = failing.predict_at([e],k)[0]
    mu = np.array(passer._spin_potentials(*args(e),k))
    repaired_mu = np.array(variants['both_algebra_repairs']._spin_potentials(*args(e),k))
    fail_mu = np.array(failing._spin_potentials(*args(e),k))
    # Evaluate all three spin states in the independent *physical* balance.
    dl,dr = ref.material_gaps(e['left_temperature'],e['right_temperature'])
    ratio = ref.DENSITY/(ref.POLARIZATION_TIME*ref.RATE*k)
    balance={}
    for name,state in [('passer',mu),('repaired_failure',repaired_mu),('original_failure',fail_mu)]:
        values = ref.transitions(*args(e),state)
        losses = np.array([ref.relaxation_integral(t,d,m)[0]
                           for t,d,m in zip((e['left_temperature'],e['right_temperature']),
                                            (dl,dr),state)])
        balance[name] = float(np.max(abs(values[3:5]-ratio*losses)))
    report['independent_reference'].append(dict(input=e,conductance=k,
        expected=float(exact['heat']),passer=float(p),repaired_failure=float(repaired),
        original_failure=float(fail),expected_polarization=np.asarray(exact['polarization']).tolist(),
        passer_polarization=mu.tolist(),repaired_failure_polarization=repaired_mu.tolist(),
        original_failure_polarization=fail_mu.tolist(),physical_dimensionless_balance_residual=balance))
    if i%10==0: print('INDEPENDENT',i+1,'of',len(checks),flush=True)

report['independent_reference_summary'] = {
    name:numerical_summary([r[name] for r in report['independent_reference']],
                           [r['expected'] for r in report['independent_reference']])
    for name in ('passer','repaired_failure','original_failure')}
report['maximum_physical_balance_residual'] = {
    name:max(r['physical_dimensionless_balance_residual'][name] for r in report['independent_reference'])
    for name in ('passer','repaired_failure','original_failure')}
report['maximum_polarization_error'] = {
    name:max(float(np.max(abs(np.asarray(r[name+'_polarization'])-r['expected_polarization'])))
             for r in report['independent_reference'])
    for name in ('passer','repaired_failure','original_failure')}

# Raise only Gauss order, preserving exact model equations and parameter.
refined = memory_module(passpath.read_text(),'passer_order384')
refined.NODES,refined.WEIGHTS=leggauss(384)
refinement = []
for e,k in checks:
    refinement.append(abs(refined.predict_at([e],k)[0]-passer.predict_at([e],k)[0]))
report['passer_192_to_384_order_max_absolute_pW']=float(max(refinement))
report['duration_seconds']=time.perf_counter()-start
assert all(digest(ROOT/p)==h for p,h in source_hashes.items())
OUT.write_text(json.dumps(report,indent=2)+'\n')
print('FINAL',report['independent_reference_summary'],report['maximum_physical_balance_residual'],
      report['maximum_polarization_error'],report['passer_192_to_384_order_max_absolute_pW'],
      report['duration_seconds'],flush=True)
