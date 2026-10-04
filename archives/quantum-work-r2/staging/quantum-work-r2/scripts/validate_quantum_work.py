"""Validate the staged maximally mixed fourth-cumulant work revision."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/quantum-work'
KEYS = ['amplitude_a','amplitude_b','phase','time_a','time_b']


def load(path,name):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def physical_checks(good,bad,ref):
    rng = np.random.default_rng(190101)
    identities,unitarity,gauges,refs,refinements = [],[],[],[],[]
    for i in range(48):
        scale = rng.uniform(.8,1.2)
        args = rng.uniform([-1.1,-1.1,-np.pi,0,0],[1.1,1.1,np.pi,1.3,1.3]).tolist()
        u = good.evolution(scale,*args)
        a = scale*good.D
        b = u.conj().T@a@u
        comm = a@b-b@a
        exact,source = good.statistics(*args,scale),bad.statistics(*args,scale)
        identities.extend(abs(exact[:3]-source[:3]))
        identities.append(abs(exact[3]-source[3]-np.vdot(comm,comm).real/3))
        unitarity.append(float(abs(u.conj().T@u-np.eye(3)).max()))
        shift = a+.73*np.eye(3)
        amp1,amp2,phase,t1,t2 = args
        shifted_u = expm(-1j*t2*(shift+amp2*(np.cos(phase)*good.X+np.sin(phase)*good.Y)))@expm(-1j*t1*(shift+amp1*good.X))
        gauges.append(float(abs(shifted_u.conj().T@shift@shifted_u-shift-(b-a)).max()))
        if i<16:
            independent = ref.cumulants(*args,scale)
            refined = ref.cumulants(*args,scale,.24,96)
            refs.append(float(abs(independent-exact).max()))
            refinements.append(float(abs(independent-refined).max()))
    assert max(identities+unitarity+gauges)<1e-11
    assert max(refs+refinements)<1e-8
    zeros = []
    for s in [.8,1.06,1.2]:
        for controls in [(0.,0.,.8,.9,1.2),(.6,.7,.3,0.,0.)]:
            zeros.extend(abs(good.statistics(*controls,s)))
            zeros.extend(abs(bad.statistics(*controls,s)))
            zeros.extend(abs(ref.cumulants(*controls,s)))
    assert max(zeros)<1e-8
    calibration = ref.calibration_inputs()
    grid = np.linspace(.8,1.2,201)
    equivalence = max(float(abs(good.predict_at(calibration,s)-bad.predict_at(calibration,s)).max()) for s in grid[::20])
    derivatives = np.array([(good.predict_at(calibration,s+1e-5)-good.predict_at(calibration,s-1e-5))/2e-5 for s in grid])
    minimum_derivative = float(derivatives[:,::2].min())
    third_minimum = min(float(abs(good.predict_at(calibration,s)[1::2]).min()) for s in grid)
    assert equivalence<1e-12 and minimum_derivative>.2 and third_minimum>.008
    profiles, recovery = [],[]
    for s in [.8,.9,1.06,1.1,1.2]:
        exact = good.predict_at(calibration,s)
        losses = np.array([np.sum((good.predict_at(calibration,z)-exact)**2) for z in grid])
        ix = int(losses.argmin())
        assert np.all(np.diff(losses[:ix+1])<0) and np.all(np.diff(losses[ix:])>0)
        records = [dict(input=e,value=float(v),sigma=.0003) for e,v in zip(calibration,exact)]
        error = abs(good.Model().fit(records).energy_scale/s-1)
        assert error<1e-7
        profiles.append(dict(true_scale=s,minimum=float(grid[ix]),strictly_decreasing_then_increasing=True))
        recovery.append(error)
    hidden = ref.hidden_inputs()
    ranges = {}
    for name,inputs in hidden.items():
        if name=='lower_cumulants': continue
        values = np.array([good.predict_at(inputs,s) for s in grid])
        source = np.array([bad.predict_at(inputs,s) for s in grid])
        separation = np.sqrt(np.mean((values-source)**2,axis=1)/np.mean(values**2,axis=1))
        assert values.max()<0 and source.max()<0
        assert abs(values).min()>.5 and separation.min()>.1
        ranges[name] = dict(minimum_absolute_fourth=float(abs(values).min()),minimum_relative_shortcut_error=float(separation.min()))
    inputs = sum(hidden.values(),[])
    ref.propagate.cache_clear();ref.cumulants.cache_clear()
    start = time.perf_counter()
    truth = ref.predict(inputs)
    runtime = time.perf_counter()-start
    error = float(abs(truth-good.predict_at(inputs,ref.TRUE_PARAMETER)).max())
    assert error<1e-8 and runtime<30
    for e in calibration+inputs:
        assert set(e)==set(KEYS+['cumulant'])
        assert -1.1<=e['amplitude_a']<=1.1 and -1.1<=e['amplitude_b']<=1.1
        assert -np.pi<=e['phase']<=np.pi and 0<=e['time_a']<=1.3 and 0<=e['time_b']<=1.3
        assert e['cumulant'] in [1,2,3,4]
    return dict(first_three_and_fourth_commutator_identity=max(identities),unitarity=max(unitarity),
                energy_shift_invariance=max(gauges),independent_reference=max(refs),reference_refinement=max(refinements),
                zero_work_limits=max(zeros),calibration_equivalence=equivalence,
                minimum_variance_derivative=minimum_derivative,minimum_absolute_calibration_third=third_minimum,
                identifiability_profiles=profiles,noiseless_relative_recovery=max(recovery),
                hidden_full_scale_ranges=ranges,hidden_reference_error=error,cold_hidden_reference_seconds=runtime,
                input_domain_verified=True)


def local_controls():
    output = {}
    for label,path in [('oracle',TASK/'solution/model.py'),('shortcut',ROOT/'scripts/quantum_work_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='quantum-work-r2-control-') as directory:
            work = Path(directory)
            shutil.copytree(TASK/'environment',work/'app')
            shutil.copytree(TASK/'tests',work/'tests')
            shutil.copy2(path,work/'app/model.py')
            env = dict(os.environ,PYTHONPATH=str(work/'app'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            start = time.perf_counter()
            run = subprocess.run([sys.executable,'-m','pytest','-q',str(work/'app/test_public.py'),str(work/'tests/test_hidden.py')],
                                 cwd=work,env=env,capture_output=True,text=True)
            output[label] = dict(returncode=run.returncode,seconds=time.perf_counter()-start,stdout=run.stdout,stderr=run.stderr)
            assert run.returncode==(0 if label=='oracle' else 1)
            assert ('8 passed' in run.stdout if label=='oracle' else '3 failed, 5 passed' in run.stdout)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate',action='store_true')
    args = parser.parse_args()
    started = time.time()
    good = load(TASK/'solution/model.py','oracle')
    bad = load(ROOT/'scripts/quantum_work_baseline.py','shortcut')
    ref = load(TASK/'tests/reference.py','reference')
    meta = json.loads((TASK/'tests/metadata.json').read_text())
    sigma,true = meta['measurement_sigma'],ref.TRUE_PARAMETER
    inputs = ref.calibration_inputs()
    exact = ref.predict(inputs)
    if args.generate:
        measured = exact+np.random.default_rng(meta['calibration_seed']).normal(0,sigma,len(inputs))
        records = [dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,measured)]
        for name in ['environment','tests']:
            (TASK/name/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
    records = json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records==json.loads((TASK/'tests/data/calibration.json').read_text())
    assert [r['input'] for r in records]==inputs and all(r['sigma']==sigma for r in records)
    hidden = ref.hidden_inputs()
    truth = {name:ref.predict(es) for name,es in hidden.items()}
    diagnostic = [name for name in hidden if name!='lower_cumulants']
    def score(model):
        return {name:float(np.sqrt(np.mean((model.predict(es)-truth[name])**2)/np.mean(truth[name]**2))) for name,es in hidden.items()}
    def chi2(model,rows):
        return float(np.sum(((model.predict(inputs)-np.array([r['value'] for r in rows]))/sigma)**2)/(len(rows)-1))
    report = dict(task='quantum-work',revision=2,status='staged_unevaluated',calibration_seed=meta['calibration_seed'],noise_seed=meta['noise_seed'],controls={})
    for label,module in [('oracle',good),('shortcut',bad)]:
        model = module.Model().fit(records)
        result = dict(energy_scale=model.energy_scale,parameter_relative_error=abs(model.energy_scale/true-1),calibration_chi2=chi2(model,records),hidden=score(model))
        assert result['calibration_chi2']<1.5 and result['parameter_relative_error']<.03
        assert result['hidden']['lower_cumulants']<meta['prediction_limit']
        assert (max(result['hidden'].values())<meta['prediction_limit'] if label=='oracle' else min(result['hidden'][k] for k in diagnostic)>meta['prediction_limit'])
        report['controls'][label] = result
    rng = np.random.default_rng(meta['noise_seed'])
    parameters,chis,oracle_errors,shortcut_errors = [],[],[],[]
    for _ in range(256):
        rows = [dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,exact+rng.normal(0,sigma,len(inputs)))]
        a,b = good.Model().fit(rows),bad.Model().fit(rows)
        assert abs(a.energy_scale-b.energy_scale)<1e-7
        parameters.append(a.energy_scale);chis.append(chi2(a,rows))
        oracle_errors.append(max(score(a).values()))
        scores = score(b);shortcut_errors.append(min(scores[k] for k in diagnostic))
        assert chis[-1]<1.5 and abs(a.energy_scale/true-1)<.03
        assert oracle_errors[-1]<meta['prediction_limit'] and shortcut_errors[-1]>meta['prediction_limit']
        assert scores['lower_cumulants']<meta['prediction_limit']
    report['noise'] = dict(realizations=256,calibration_passes=256,parameter_passes=256,oracle_passes=256,shortcut_rejections=256,
                           maximum_chi2=max(chis),maximum_relative_parameter_error=max(abs(np.array(parameters)/true-1)),
                           parameter_range=[min(parameters),max(parameters)],oracle_hidden_max=max(oracle_errors),shortcut_hidden_min=min(shortcut_errors))
    report['physical_checks'] = physical_checks(good,bad,ref)
    report['local_controls'] = local_controls()
    report['seconds'] = time.time()-started
    report['source_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TASK/'instruction.md',TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',TASK/'tests/metadata.json',TASK/'environment/data/calibration.json',ROOT/'scripts/quantum_work_baseline.py',Path(__file__)]}
    destination = ROOT/'results/quantum-work-r2-validation.json'
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
