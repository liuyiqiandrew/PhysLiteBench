"""Scientific validation and local controls for staged spin-body torque."""
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

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/spin-body-torque'


def load(path,name):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def physical_checks(good,bad,ref):
    maximum_normalization = maximum_balance = maximum_dc = maximum_heat = 0.
    maximum_eigenvalue = -np.inf
    minimum_z = 1.
    for bias in np.linspace(.8,1.4,11):
        for frequency in np.linspace(.4,1.4,11):
            for amplitude in [.005,.015,.025,.04]:
                m = good.stationary_orientation(bias,frequency,amplitude)
                minimum_z = min(minimum_z,m[2])
                maximum_normalization = max(maximum_normalization,abs(m@m-1))
                v = frequency*np.array([-m[1],m[0],0.])
                field = np.array([amplitude,0.,bias])
                gilbert_residual = v+np.cross(m,field)-.18*np.cross(m,v)
                maximum_balance = max(maximum_balance,float(abs(gilbert_residual).max()))
                torque = .18*np.cross(m,v)
                external = np.cross(m,field)
                maximum_dc = max(maximum_dc,abs(torque[2]-external[2]))
                absorbed = -m@np.array([0.,frequency*amplitude,0.])
                heat = .18*(v@v)
                maximum_heat = max(maximum_heat,abs(absorbed-heat))
                assert heat>0
                def drift(xy):
                    full = np.array([*xy,np.sqrt(1-xy@xy)])
                    return (ref.laboratory_derivative(0.,full,bias,frequency,amplitude)-frequency*np.array([-full[1],full[0],0.]))[:2]
                h = 1e-6
                jacobian = np.column_stack([(drift(m[:2]+h*np.eye(2)[j])-drift(m[:2]-h*np.eye(2)[j]))/(2*h) for j in range(2)])
                maximum_eigenvalue = max(maximum_eigenvalue,float(np.linalg.eigvals(jacobian).real.max()))
    assert maximum_normalization<1e-12 and maximum_balance<1e-12 and maximum_dc<1e-12
    assert maximum_heat<1e-12 and minimum_z>.95 and maximum_eigenvalue<-.1
    rng = np.random.default_rng(319137)
    reference_errors,refinement_errors,period_errors = [],[],[]
    cases = [ref.experiment(.8,.8,.04,.5),ref.experiment(1.4,.4,.04,-2.),ref.experiment(.8,1.4,.04,2.)]
    cases += [ref.experiment(rng.uniform(.8,1.4),rng.uniform(.4,1.4),rng.uniform(.005,.04),rng.uniform(-np.pi,np.pi)) for _ in range(12)]
    for e in cases:
        predicted = good.torque_at(e,ref.TRUE_PARAMETER)
        reference = ref.torque(e)
        refined = ref.torque(e,decays=38,rtol=2e-12)
        reference_errors.append(float(abs(predicted-reference).max()))
        refinement_errors.append(float(abs(reference-refined).max()))
        solution,end,period = ref.periodic_solution(e['bias'],e['frequency'],e['amplitude'])
        period_errors.append(float(abs(solution(end)-solution(end-period)).max()))
    assert max(reference_errors+refinement_errors+period_errors)<1e-8
    calibration = ref.calibration_inputs()
    design = good.predict_at(calibration,1.)
    source_design = bad.predict_at(calibration,1.)
    assert abs(design-source_design).max()<1e-13 and design.min()>0
    recovery = []
    for moment in [.8,1.,1.07,1.2]:
        records = [dict(input=e,value=float(y),sigma=.00001) for e,y in zip(calibration,moment*design)]
        recovery.append(abs(good.Model().fit(records).moment-moment))
    assert max(recovery)<1e-12
    hidden = ref.hidden_inputs()
    inputs = sum(hidden.values(),[])
    ref.periodic_solution.cache_clear()
    started = time.perf_counter()
    truth = ref.predict(inputs)
    runtime = time.perf_counter()-started
    difference = abs(truth-good.predict_at(inputs,ref.TRUE_PARAMETER)).max()
    assert difference<1e-8 and runtime<40
    minimum_signal = min(float(abs(ref.predict(es)).min()) for name,es in hidden.items() if name!='longitudinal_checks')
    assert minimum_signal>.0005
    for e in calibration+inputs:
        assert .8<=e['bias']<=1.4 and .4<=e['frequency']<=1.4 and .005<=e['amplitude']<=.04
        assert -np.pi<=e['phase']<=np.pi and e['component'] in ['x','y','z']
    return dict(grid_points=484,minimum_north_branch_z=float(minimum_z),maximum_stability_eigenvalue=maximum_eigenvalue,
                orientation_normalization=maximum_normalization,stationary_Gilbert_residual=maximum_balance,
                dc_torque_equivalence=maximum_dc,energy_balance=maximum_heat,
                nonlinear_reference_max=max(reference_errors),reference_refinement_max=max(refinement_errors),
                periodic_convergence_max=max(period_errors),calibration_equivalence=float(abs(design-source_design).max()),
                minimum_calibration_design=float(design.min()),weighted_fit_curvature=float(2*np.sum((design/.00001)**2)),
                noiseless_absolute_parameter_recovery=max(recovery),hidden_reference_error=float(difference),
                cold_hidden_reference_seconds=runtime,minimum_absolute_hidden_transverse_signal=minimum_signal,
                public_domain_verified=True)


def local_controls():
    report = {}
    for label,path in [('oracle',TASK/'solution/model.py'),('shortcut',ROOT/'scripts/spin_body_torque_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='spin-body-torque-control-') as tmp:
            work = Path(tmp)
            shutil.copytree(TASK/'environment',work/'app')
            shutil.copytree(TASK/'tests',work/'tests')
            shutil.copy2(path,work/'app/model.py')
            env = dict(os.environ,PYTHONPATH=str(work/'app'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            started = time.perf_counter()
            result = subprocess.run([sys.executable,'-m','pytest','-q',str(work/'app/test_public.py'),str(work/'tests/test_hidden.py')],cwd=work,env=env,capture_output=True,text=True)
            report[label] = dict(returncode=result.returncode,seconds=time.perf_counter()-started,stdout=result.stdout,stderr=result.stderr)
            assert result.returncode==(0 if label=='oracle' else 1)
            assert ('8 passed' in result.stdout if label=='oracle' else '3 failed, 5 passed' in result.stdout)
    return report


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');args = parser.parse_args()
    started = time.time()
    good = load(TASK/'solution/model.py','oracle');bad = load(ROOT/'scripts/spin_body_torque_baseline.py','shortcut');ref = load(TASK/'tests/reference.py','reference')
    meta = json.loads((TASK/'tests/metadata.json').read_text());sigma = meta['measurement_sigma'];true = ref.TRUE_PARAMETER
    inputs = ref.calibration_inputs();exact = ref.predict(inputs)
    if args.generate:
        measured = exact+np.random.default_rng(meta['calibration_seed']).normal(0,sigma,len(inputs))
        records = [dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,measured)]
        for name in ['environment','tests']:(TASK/name/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
    records = json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records==json.loads((TASK/'tests/data/calibration.json').read_text())
    assert [r['input'] for r in records]==inputs and all(r['sigma']==sigma for r in records)
    hidden = ref.hidden_inputs();truth = {name:ref.predict(es) for name,es in hidden.items()}
    diagnostics = [name for name in hidden if name!='longitudinal_checks']
    def score(model):return {name:float(np.sqrt(np.mean((model.predict(es)-truth[name])**2)/np.mean(truth[name]**2))) for name,es in hidden.items()}
    def chi(model,rows):return float(np.sum(((model.predict(inputs)-np.array([r['value'] for r in rows]))/sigma)**2)/(len(rows)-1))
    report = dict(task='spin-body-torque',revision=1,status='staged_unevaluated',calibration_seed=meta['calibration_seed'],noise_seed=meta['noise_seed'],controls={})
    for label,module in [('oracle',good),('shortcut',bad)]:
        model = module.Model().fit(records);scores = score(model)
        result = dict(moment=model.moment,parameter_relative_error=abs(model.moment/true-1),calibration_chi2=chi(model,records),hidden=scores)
        assert result['calibration_chi2']<1.5 and result['parameter_relative_error']<.03 and scores['longitudinal_checks']<.04
        assert (max(scores.values())<.04 if label=='oracle' else min(scores[k] for k in diagnostics)>.04)
        report['controls'][label] = result
    rng = np.random.default_rng(meta['noise_seed']);parameters=[];chis=[];oracle_errors=[];shortcut_errors=[]
    for _ in range(256):
        rows = [dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,exact+rng.normal(0,sigma,len(inputs)))]
        a,b = good.Model().fit(rows),bad.Model().fit(rows)
        assert abs(a.moment-b.moment)<1e-12
        parameters.append(a.moment);chis.append(chi(a,rows));oracle_errors.append(max(score(a).values()))
        scores = score(b);shortcut_errors.append(min(scores[k] for k in diagnostics))
        assert chis[-1]<1.5 and abs(a.moment/true-1)<.03 and oracle_errors[-1]<.04 and shortcut_errors[-1]>.04
        assert scores['longitudinal_checks']<.04
    report['noise'] = dict(realizations=256,calibration_passes=256,parameter_passes=256,oracle_passes=256,shortcut_rejections=256,
        maximum_chi2=max(chis),maximum_relative_parameter_error=float(abs(np.array(parameters)/true-1).max()),
        parameter_range=[min(parameters),max(parameters)],oracle_hidden_max=max(oracle_errors),shortcut_hidden_min=min(shortcut_errors))
    report['physical_checks'] = physical_checks(good,bad,ref)
    report['local_controls'] = local_controls();report['seconds'] = time.time()-started
    report['source_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TASK/'instruction.md',TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',TASK/'tests/metadata.json',TASK/'environment/data/calibration.json',ROOT/'scripts/spin_body_torque_baseline.py',Path(__file__)]}
    (ROOT/'results/spin-body-torque-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
