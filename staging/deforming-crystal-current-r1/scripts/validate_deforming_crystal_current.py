"""Scientific validation and isolated completed controls, without model evaluation."""
import argparse
import ast
import hashlib
import importlib.util
from itertools import product
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import numpy as np

BASE = Path(__file__).resolve().parents[1]
TASK = BASE/'tasks/deforming-crystal-current'
RESULTS = BASE/'results'
REPORT = {}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle = load('oracle', TASK/'solution/model.py')
source = load('source', BASE/'scripts/deforming_crystal_current_baseline.py')
reference = load('reference', TASK/'tests/reference.py')
metadata = json.loads((TASK/'tests/metadata.json').read_text())


def error(actual, expected):
    return float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))


def local_controls():
    controls = {}
    for name, model in [('oracle', TASK/'solution/model.py'),
                        ('shortcut', BASE/'scripts/deforming_crystal_current_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='deforming-crystal-') as directory:
            path = Path(directory)
            shutil.copytree(TASK/'environment', path/'app')
            shutil.copytree(TASK/'tests', path/'tests')
            shutil.copyfile(model, path/'app/model.py')
            env = os.environ.copy()
            env.update(PYTHONPATH=str(path/'app'), PYTHONDONTWRITEBYTECODE='1',
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
            start = time.monotonic()
            result = subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
                                     str(path/'app/test_public.py'),str(path/'tests/test_hidden.py')],
                                    cwd=path,env=env,capture_output=True,text=True,timeout=60)
            controls[name] = {'returncode':result.returncode,'seconds':time.monotonic()-start,
                              'stdout':result.stdout,'stderr':result.stderr}
    (RESULTS/'deforming-crystal-current-r1-local-controls.json').write_text(json.dumps(controls,indent=2)+'\n')
    assert controls['oracle']['returncode']==0 and '7 passed' in controls['oracle']['stdout']
    assert controls['shortcut']['returncode']==1 and '3 failed, 4 passed' in controls['shortcut']['stdout']
    return {name:{key:value for key,value in row.items() if key not in ['stdout','stderr']}
            for name,row in controls.items()}


def run(generate):
    start = time.monotonic()
    stiffness = reference.TRUE_PARAMETER
    sigma = metadata['measurement_sigma']
    inputs = reference.calibration_inputs()*metadata['calibration_repeats']
    clean = reference.predict(inputs, stiffness)
    if generate:
        rng = np.random.default_rng(metadata['calibration_seed'])
        values = clean+rng.normal(0,sigma,len(clean))
        records = [{'input':e,'value':float(v),'sigma':sigma} for e,v in zip(inputs,values)]
        content = json.dumps(records,indent=2)+'\n'
        for side in ['environment','tests']:
            (TASK/side/'data/calibration.json').write_text(content)
    records = json.loads((TASK/'environment/data/calibration.json').read_text())
    assert [r['input'] for r in records]==inputs
    assert all(set(r)=={'input','value','sigma'} and r['sigma']==sigma for r in records)
    assert (TASK/'environment/data/calibration.json').read_bytes()==(TASK/'tests/data/calibration.json').read_bytes()
    REPORT.update(status='science_in_progress',model_evaluations=0,
                  calibration_records=len(records),calibration_distinct_settings=len(reference.calibration_inputs()),
                  sigma=sigma,true_stiffness=stiffness,prediction_limit=.04)
    groups = reference.hidden_inputs()
    truths = {name:reference.predict(experiments,stiffness) for name,experiments in groups.items()}
    actual = {}
    for name,module in [('oracle',oracle),('shortcut',source)]:
        model = module.Model().fit(records)
        residual = (model.predict(inputs)-[r['value'] for r in records])/sigma
        actual[name] = {'parameter':model.stiffness,'parameter_relative_error':abs(model.stiffness/stiffness-1),
                        'chi2':float(residual@residual/(len(records)-1)),
                        'hidden':{group:error(model.predict(experiments),truths[group])
                                  for group,experiments in groups.items()}}
    REPORT['actual_data'] = actual
    REPORT['scored_reference'] = {
        'count':sum(map(len,groups.values())),
        'max_error':max(float(np.max(abs(oracle.predict_at(e,stiffness)-truths[g]))) for g,e in groups.items()),
        'max_step_grid_refinement':max(float(np.max(abs(reference.predict(e,stiffness,1e-4,9,4096)-truths[g])))
                                       for g,e in groups.items()),
        'max_cell_count_change':max(float(np.max(abs(reference.predict(e,stiffness,2e-4,15,4096)-truths[g])))
                                   for g,e in groups.items()),
        'min_physical_diagnostic_signal':float(min(np.min(truths[g]) for g in groups if g!='longitudinal_anchor')),
        'min_source_diagnostic_signal':float(min(np.min(source.predict_at(e,stiffness)) for g,e in groups.items() if g!='longitudinal_anchor')),
        'calibration_reference_error_sigma':float(np.max(abs(oracle.predict_at(inputs,stiffness)-clean))/sigma)}
    max_chi = max_parameter = max_oracle = max_anchor = 0.
    min_source = float('inf')
    fit_values = []
    noise_rows = []
    rng = np.random.default_rng(metadata['noise_seed'])
    for trial in range(256):
        values = clean+rng.normal(0,sigma,len(clean))
        sample = [{'input':e,'value':float(v),'sigma':sigma} for e,v in zip(inputs,values)]
        row = {'draw':trial}
        for name,module in [('oracle',oracle),('shortcut',source)]:
            model = module.Model().fit(sample)
            fit_values.append(model.stiffness)
            chi = float(np.sum(((model.predict(inputs)-values)/sigma)**2)/(len(inputs)-1))
            parameter_error = abs(model.stiffness/stiffness-1)
            errors = {g:error(model.predict(e),truths[g]) for g,e in groups.items()}
            row[name] = {'parameter':model.stiffness,'chi2':chi,'errors':errors}
            max_chi = max(max_chi,chi);max_parameter = max(max_parameter,parameter_error)
            assert chi<1.5 and parameter_error<.03
            if name=='oracle':
                max_oracle = max(max_oracle,max(errors.values()))
                assert max(errors.values())<.04
            else:
                max_anchor = max(max_anchor,errors['longitudinal_anchor'])
                min_source = min(min_source,min(v for g,v in errors.items() if g!='longitudinal_anchor'))
                assert errors['longitudinal_anchor']<.04
                assert min(v for g,v in errors.items() if g!='longitudinal_anchor')>.04
        noise_rows.append(row)
    REPORT['noise'] = {'trials':256,'all_expected_outcomes':True,'max_chi2':max_chi,
                      'max_parameter_error':max_parameter,'max_oracle_error':max_oracle,
                      'min_shortcut_diagnostic_error':min_source,'max_shortcut_anchor_error':max_anchor,
                      'fit_min':min(fit_values),'fit_max':max(fit_values)}
    (RESULTS/'deforming-crystal-current-r1-noise-checks.json').write_text(json.dumps(noise_rows,indent=2)+'\n')
    unique = reference.calibration_inputs()
    grid = np.linspace(.8,1.2,201)
    grid_predictions = np.array([oracle.predict_at(unique,k) for k in grid])
    max_fit = max_equality = 0.
    min_physical = min_source_signal = float('inf')
    gaps = {g:float('inf') for g in groups if g!='longitudinal_anchor'}
    for k in np.linspace(.8,1.2,41):
        values = oracle.predict_at(inputs,k)
        max_equality = max(max_equality,float(np.max(abs(source.predict_at(inputs,k)-values))))
        sample = [{'input':e,'value':float(v),'sigma':sigma} for e,v in zip(inputs,values)]
        for module in [oracle,source]:
            max_fit = max(max_fit,abs(module.Model().fit(sample).stiffness-k))
        profile = np.sum((grid_predictions-oracle.predict_at(unique,k))**2,axis=1)
        index = int(round((k-.8)/.4*200))
        assert np.argmin(profile)==index
        assert np.all(np.diff(profile[:index+1])<0) and np.all(np.diff(profile[index:])>0)
        for group,experiments in groups.items():
            if group!='longitudinal_anchor':
                correct = oracle.predict_at(experiments,k)
                wrong = source.predict_at(experiments,k)
                gaps[group] = min(gaps[group],error(wrong,correct))
                min_physical = min(min_physical,float(np.min(correct)))
                min_source_signal = min(min_source_signal,float(np.min(wrong)))
    assert max_fit<1e-7 and max_equality<1e-12 and min(gaps.values())>.04
    assert min_physical>.1 and min_source_signal>.02
    REPORT['identifiability'] = {
        'noiseless_parameter_samples':41,'profile_points':201,'max_recovery_error':max_fit,
        'max_calibration_equivalence':max_equality,'unique_profile_minima':True,
        'proof':'For longitudinal calibration, x is axial and h=.35(l^2-1). The current .7l/(K+12x^2) strictly decreases in K because d(K+12x^2)/dK=(K-12x^2)/(K+12x^2)>0; |x|<=.1113 implies12x^2<.149<K. Signed reversed drives have identical squared fitting residuals. All noiseless residual derivatives have the same sign away from truth.',
        'minimum_group_gaps':gaps,'min_diagnostic_physical_signal_over_K':min_physical,
        'min_diagnostic_source_signal_over_K':min_source_signal,
        'qualification':'Global injectivity and noiseless profiles do not assert every arbitrary noisy objective is unimodal.'}
    max_ref = max_refinement = max_charge_balance = max_state = max_source_derivative = 0.
    min_cell_margin = min_hessian = min_determinant = float('inf')
    basis = []
    for i in range(3):
        for j in range(i,3):
            g = np.zeros((3,3));g[i,j] = 1
            basis.append(g)
    corners = [reference.experiment(diagonal,shears,np.zeros((3,3)))
               for diagonal in product([.88,1.12],repeat=3)
               for shears in product([-.12,.12],repeat=3)]
    cases = []
    for k in [.8,1.,1.2]:
        for point in corners:
            for g in basis:
                for sign in [-1,1]:
                    cases.append((k,{'deformation':point['deformation'],'drive':(sign*g).tolist()}))
    rng = np.random.default_rng(428719)
    for _ in range(64):
        e = reference.experiment(rng.uniform(.88,1.12,3),rng.uniform(-.12,.12,3),
                                np.triu(rng.uniform(-1,1,(3,3))))
        cases.append((rng.uniform(.8,1.2),e))
    for k,e in cases:
        f,g = np.asarray(e['deformation']),np.asarray(e['drive'])
        correct = oracle.predict_at([e],k)[0]
        ref = reference.predict([e],k)[0]
        fine = reference.predict([e],k,1e-4,9,4096)[0]
        max_ref = max(max_ref,abs(correct-ref))
        max_refinement = max(max_refinement,abs(ref-fine))
        s = oracle.internal_state(f,k)
        max_state = max(max_state,float(np.max(abs(s-reference.equilibrium(f,k)))))
        center = np.array([.25,.25,.20])+s
        min_cell_margin = min(min_cell_margin,float(np.min(center)),float(np.min(1-center)))
        x = s-oracle.S0
        min_hessian = min(min_hessian,float(np.linalg.eigvalsh((k+4*x@x)*np.eye(3)+8*np.outer(x,x)).min()))
        min_determinant = min(min_determinant,float(np.linalg.det(f)))
        max_charge_balance = max(max_charge_balance,abs(reference.electrode_charge(f,k)+reference.electrode_charge(f,k,electrode='lower')))
        def p_at(t):
            a = f+t*g
            return -a@reference.equilibrium(a,k)/np.linalg.det(a)
        step = 2e-4
        dp = (p_at(-2*step)-8*p_at(-step)+8*p_at(step)-p_at(2*step))/(12*step)
        face = np.linalg.det(f)*np.linalg.solve(f.T,[0.,0.,1.])
        max_source_derivative = max(max_source_derivative,abs(source.predict_at([e],k)[0]+face@dp))
    assert max(max_ref,max_refinement,max_source_derivative)<1e-7
    assert max_state<1e-10 and max_charge_balance<1e-10 and min_cell_margin>0 and min_hessian>=.8
    REPORT['domain'] = {
        'deformation_box_corners':64,'stiffness_values':3,'signed_basis_drives':12,
        'structured_response_cases':2304,'random_interior_response_cases':64,
        'max_reference_error':max_ref,'max_step_grid_refinement':max_refinement,
        'max_independent_internal_state_error':max_state,'max_total_electrode_charge_error':max_charge_balance,
        'max_source_vs_independent_density_derivative_error':max_source_derivative,
        'sampled_minimum_cell_boundary_margin':min_cell_margin,'minimum_internal_hessian_eigenvalue':min_hessian,
        'minimum_determinant':min_determinant,
        'scope':'Every F box corner and all signed rate basis directions at three K values, plus64 full interior F/G/K controls. Response is linear in G. Numerical sampling is not an interval error proof.',
        'global_stability_branch_proof':'Hessian>=K I everywhere. For the entire F box, |h|<.223, so |x|<.279; c+s0=(.33,.31,.42) then keeps every center strictly in its selected cell. Upper triangular F has positive determinant. No metastable or coordinate-wrapping branch is selected.'}
    max_reverse = max_linear = max_frame = max_rotation = 0.
    for _ in range(64):
        k = rng.uniform(.8,1.2)
        e = reference.experiment(rng.uniform(.88,1.12,3),rng.uniform(-.12,.12,3),np.triu(rng.uniform(-1,1,(3,3))))
        f,g = np.asarray(e['deformation']),np.asarray(e['drive'])
        g2 = np.triu(rng.uniform(-1,1,(3,3)))
        q,_ = np.linalg.qr(rng.normal(size=(3,3)))
        if np.linalg.det(q)<0:q[:,0]*=-1
        for module in [oracle,source]:
            def predict(a,b):return module.predict_at([{'deformation':a.tolist(),'drive':b.tolist()}],k)[0]
            max_reverse = max(max_reverse,abs(predict(f,g)+predict(f,-g)))
            max_linear = max(max_linear,abs(predict(f,.4*g+.6*g2)-.4*predict(f,g)-.6*predict(f,g2)))
            max_frame = max(max_frame,abs(predict(q@f,q@g)-predict(f,g)))
        omega = np.array([[0.,.2,.6],[-.2,0.,-.3],[-.6,.3,0.]])
        max_rotation = max(max_rotation,abs(oracle.predict_at([{'deformation':f.tolist(),'drive':(omega@f).tolist()}],k)[0]))
    assert max(max_reverse,max_linear,max_frame,max_rotation)<1e-12
    REPORT['limits'] = {'cases':64,'max_signed_drive_reversal_error':max_reverse,'max_rate_linearity_error':max_linear,
                        'max_fixed_frame_covariance_error':max_frame,'max_physical_rigid_rotation_response':max_rotation,
                        'scope':'Arbitrary orthogonal frame changes and rigid rotation are author-only constitutive checks outside the upper-triangular public input schema. No zero-response scored group uses rigid rotation.',
                        'electrostatic_reference_qualification':'First-moment-preserving deposition makes integrated induced charge naturally mesh-exact up to roundoff; mesh refinement is not evidence of resolving all microscopic fields. Electrode totals use the exact planar-average reduction, and the cell free energy is a specified effective input.'}
    def functions(path):
        return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef)}
    assert functions(TASK/'environment/model.py')==functions(BASE/'scripts/deforming_crystal_current_baseline.py')
    REPORT['local_controls'] = local_controls()
    REPORT['status'] = 'science_and_local_controls_complete'
    REPORT['seconds'] = time.monotonic()-start
    REPORT['source_hashes'] = {str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',
                                        TASK/'tests/reference.py',TASK/'tests/test_hidden.py',BASE/'scripts/deforming_crystal_current_baseline.py',Path(__file__)]}
    (RESULTS/'deforming-crystal-current-r1-validation.json').write_text(json.dumps(REPORT,indent=2)+'\n')
    print(json.dumps(REPORT,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate',action='store_true',help='Intentionally regenerate both calibration copies.')
    try:
        run(parser.parse_args().generate)
    except Exception:
        REPORT['status'] = 'author_check_failed'
        REPORT['traceback'] = traceback.format_exc()
        (RESULTS/f'author-failure-{time.time_ns()}.json').write_text(json.dumps(REPORT,indent=2)+'\n')
        raise
