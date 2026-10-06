"""Package science and local controls; no model or Docker evaluation."""
import ast
from datetime import datetime, timezone
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
import traceback
import numpy as np

BASE = Path(__file__).resolve().parents[1]
TASK = BASE/'tasks/deforming-crystal-current'
RESULTS = BASE/'results'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle = load('oracle', TASK/'solution/model.py')
source = load('source', BASE/'scripts/deforming_crystal_current_baseline.py')
reference = load('reference', TASK/'tests/reference.py')
metadata = json.loads((TASK/'tests/metadata.json').read_text())


def error(a, b):
    return float(np.linalg.norm(np.asarray(a)-b)/np.linalg.norm(b))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_controls():
    result = {}
    for name, model in [('oracle', TASK/'solution/model.py'),
                        ('shortcut', BASE/'scripts/deforming_crystal_current_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='crystal-r2-') as directory:
            p = Path(directory)
            shutil.copytree(TASK/'environment', p/'app')
            shutil.copytree(TASK/'tests', p/'tests')
            shutil.copy2(model, p/'app/model.py')
            env = os.environ.copy()
            env.update(PYTHONPATH=str(p/'app'), PYTHONDONTWRITEBYTECODE='1',
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
            start = time.perf_counter()
            run = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                                  str(p/'app/test_public.py'),str(p/'tests/test_hidden.py')],
                                 cwd=p, env=env, capture_output=True, text=True, timeout=60)
            result[name] = {'returncode':run.returncode, 'seconds':time.perf_counter()-start,
                            'stdout':run.stdout, 'stderr':run.stderr}
    (RESULTS/'deforming-crystal-current-r2-local-controls.json').write_text(json.dumps(result,indent=2)+'\n')
    assert result['oracle']['returncode']==0 and '7 passed' in result['oracle']['stdout']
    assert result['shortcut']['returncode']==1 and '3 failed, 4 passed' in result['shortcut']['stdout']
    return result


def validate(report):
    start = time.perf_counter()
    records = json.loads((TASK/'environment/data/calibration.json').read_text())
    old_data = BASE.parents[1]/'archives/deforming-crystal-current-r1/tasks/deforming-crystal-current/environment/data/calibration.json'
    assert (TASK/'environment/data/calibration.json').read_bytes()==old_data.read_bytes()
    assert (TASK/'tests/data/calibration.json').read_bytes()==old_data.read_bytes()
    inputs = reference.calibration_inputs()*metadata['calibration_repeats']
    assert [r['input'] for r in records]==inputs
    sigma = metadata['measurement_sigma']
    assert len(records)==288 and sigma==.001
    assert all(set(r)=={'input','value','sigma'} and r['sigma']==sigma for r in records)
    report['calibration'] = {'records':len(records),'unique_settings':18,'fixed_sigma':sigma,
        'unchanged_r1_bytes':True,'sha256':digest(old_data),'omitted_load':'ideal_short'}
    groups = reference.hidden_inputs()
    true_k = reference.TRUE_PARAMETER
    truths = {name:reference.predict(rows,true_k) for name,rows in groups.items()}
    actual = {}
    for name,module in [('oracle',oracle),('shortcut',source)]:
        m = module.Model().fit(records)
        residual = (m.predict(inputs)-[r['value'] for r in records])/sigma
        actual[name] = {'stiffness':m.stiffness,'parameter_relative_error':abs(m.stiffness/true_k-1),
                        'calibration_chi2':float(residual@residual/(len(records)-1)),
                        'hidden':{g:error(m.predict(rows),truths[g]) for g,rows in groups.items()}}
        assert actual[name]['parameter_relative_error']<.03 and actual[name]['calibration_chi2']<1.5
    assert max(actual['oracle']['hidden'].values())<.04
    assert actual['shortcut']['hidden']['short_anchor']<.04
    assert min(v for g,v in actual['shortcut']['hidden'].items() if g!='short_anchor')>.04
    report['actual_data'] = actual

    # Every graded setting is independently refined, including the exact short anchors.
    report['graded_reference_rows'] = []
    for name,rows in [('calibration',reference.calibration_inputs()),*groups.items()]:
        values = oracle.predict_at(rows,true_k)
        coarse = reference.predict(rows,true_k)
        fine = reference.predict(rows,true_k,1e-4)
        finer = reference.predict(rows,true_k,5e-5)
        for e,a,b,c,d in zip(rows,values,coarse,fine,finer):
            report['graded_reference_rows'].append({'group':name,'input':e,'oracle':float(a),
                'reference':float(b),'fine':float(c),'finer':float(d),
                'error':abs(float(a-b)),'refinement_change':abs(float(c-b)),
                'finer_change':abs(float(d-c))})
    assert max(r['error'] for r in report['graded_reference_rows'])<1e-9
    assert max(r['finer_change'] for r in report['graded_reference_rows'])<1e-9
    clean = reference.predict(inputs,true_k)
    bias = float(np.max(abs(oracle.predict_at(inputs,true_k)-clean))/sigma)
    assert bias<1e-5
    report['calibration']['reference_bias_sigma'] = bias

    # Recompute the whole approved bounded domain with the actual packaged sources.
    bounded = json.loads((BASE/'prototype/report.json').read_text())
    report['domain_rows'] = []
    for old in bounded['domain_rows']:
        e = {'deformation':old['F'],'drive':old['G'],'load':old['load']}
        k = old['K']; a = oracle.predict_at([e],k)[0]; b = source.predict_at([e],k)[0]
        ref = reference.predict([e],k)[0]
        row = {'input':e,'K':k,'kind':old['kind'],'physical':float(a),'source':float(b),
               'reference':float(ref),'error':abs(float(a-ref)),
               'physical_prototype_change':abs(float(a-old['physical'])),
               'source_prototype_change':abs(float(b-old['source']))}
        if 'refinement_change' in old:
            fine = reference.predict([e],k,1e-4)[0]
            row['refinement_change'] = abs(float(fine-ref))
        f = np.asarray(e['deformation']);state = oracle.loaded_state(f,k,e['load'])
        y,residual = reference.field_reference(f,k,e['load'])
        row.update(state_error=float(np.max(abs(state-y[:3]))), node_charge=float(y[5]+y[6]),
                   field_residual=residual, cell_center=(state+[.25,.25,.20]).tolist())
        report['domain_rows'].append(row)
    assert max(r['error'] for r in report['domain_rows'])<1e-9
    assert max(max(r['physical_prototype_change'],r['source_prototype_change']) for r in report['domain_rows'])<1e-12
    assert max(r['state_error'] for r in report['domain_rows'])<1e-10

    report['parameter_rows'] = []
    inputs_unique = reference.calibration_inputs()
    profile_k = np.linspace(.8,1.2,201)
    profiles = np.asarray([oracle.predict_at(inputs_unique,k) for k in profile_k])
    for k in np.linspace(.8,1.2,41):
        values = oracle.predict_at(inputs,k)
        sample = [{'input':e,'value':float(v),'sigma':sigma} for e,v in zip(inputs,values)]
        fits = {name:module.Model().fit(sample).stiffness for name,module in [('oracle',oracle),('shortcut',source)]}
        equality = float(np.max(abs(values-source.predict_at(inputs,k))))
        loss = np.sum((profiles-oracle.predict_at(inputs_unique,k))**2,axis=1)
        index = int(round((k-.8)/.4*200))
        assert int(np.argmin(loss))==index and np.all(np.diff(loss[:index+1])<0) and np.all(np.diff(loss[index:])>0)
        group_rows = {}
        for g,rows in groups.items():
            a = oracle.predict_at(rows,k); b = source.predict_at(rows,k); r = reference.predict(rows,k)
            group_rows[g] = {'physical':a.tolist(),'source':b.tolist(),'reference':r.tolist(),
                'source_error':error(b,r),'reference_error':float(np.max(abs(a-r)))}
        assert max(abs(v-k) for v in fits.values())<1e-7 and equality<1e-12
        assert min(v['source_error'] for g,v in group_rows.items() if g!='short_anchor')>.04
        report['parameter_rows'].append({'K':float(k),'fits':fits,'calibration_equality':equality,'groups':group_rows})

    report['noise_rows'] = []
    rng = np.random.default_rng(metadata['noise_seed'])
    for family,k_values in [('nominal',np.full(256,true_k)),('domain',np.linspace(.8,1.2,256))]:
        for index,k in enumerate(k_values):
            exact = oracle.predict_at(inputs,k)
            values = exact+rng.normal(0,sigma,len(inputs))
            sample = [{'input':e,'value':float(v),'sigma':sigma} for e,v in zip(inputs,values)]
            target = {g:oracle.predict_at(rows,k) for g,rows in groups.items()}
            row = {'family':family,'index':index,'true_K':float(k)}
            for name,module in [('oracle',oracle),('shortcut',source)]:
                m = module.Model().fit(sample)
                chi = float(np.sum(((m.predict(inputs)-values)/sigma)**2)/(len(inputs)-1))
                relative = abs(m.stiffness/k-1)
                errors = {g:error(m.predict(rows),target[g]) for g,rows in groups.items()}
                row[name] = {'fitted_K':m.stiffness,'parameter_error':float(relative),'chi2':chi,'errors':errors}
                assert relative<.03 and chi<1.5
                if name=='oracle': assert max(errors.values())<.04
                else:
                    assert errors['short_anchor']<.04
                    assert min(v for g,v in errors.items() if g!='short_anchor')>.04
            report['noise_rows'].append(row)

    # Reuse all frozen out-of-domain limit/scaling evidence, and verify the actual
    # package reproduces every corresponding g=1 allowed or large-load limit.
    report['limit_rows'] = []
    f = np.diag([.96,1.04,1.1]);f[0,1],f[0,2],f[1,2]=.04,-.03,.02
    g = np.diag([.15,.05,1.])
    for k in [.8,1.,1.2]:
        for cap in [None,.5,1.,2.,1e2,1e4,1e6]:
            e={'deformation':f.tolist(),'drive':g.tolist(),'load':cap}
            a=oracle.predict_at([e],k)[0];b=source.predict_at([e],k)[0];r=reference.predict([e],k)[0]
            report['limit_rows'].append({'K':k,'load':cap,'physical':float(a),'source':float(b),
                'reference':float(r),'reference_error':abs(float(a-r)),
                'outside_public_domain':cap is not None and cap>2})
            if cap is None:assert abs(a-b)<1e-14
    report['symmetry_rows'] = []
    rng=np.random.default_rng(613937)
    for _ in range(48):
        f=np.diag(rng.uniform(.88,1.12,3));f[0,1],f[0,2],f[1,2]=rng.uniform(-.12,.12,3)
        g=np.triu(rng.uniform(-1,1,(3,3)));g2=np.triu(rng.uniform(-1,1,(3,3)))
        k=float(rng.uniform(.8,1.2));cap=float(rng.uniform(.5,2))
        q,_=np.linalg.qr(rng.normal(size=(3,3)))
        if np.linalg.det(q)<0:q[:,0]*=-1
        omega=np.array([[0.,.2,.6],[-.2,0.,-.3],[-.6,.3,0.]])
        row={'K':k,'load':cap}
        for name,module in [('oracle',oracle),('shortcut',source)]:
            def pred(a,b):return module.predict_at([{'deformation':a.tolist(),'drive':b.tolist(),'load':cap}],k)[0]
            row[name]={'reversal':abs(float(pred(f,g)+pred(f,-g))),
                'linearity':abs(float(pred(f,.4*g+.6*g2)-.4*pred(f,g)-.6*pred(f,g2))),
                'frame':abs(float(pred(q@f,q@g)-pred(f,g))),
                'rigid_rotation':abs(float(pred(f,omega@f)))}
            assert max(row[name].values())<1e-12
        report['symmetry_rows'].append(row)

    def functions(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    assert functions(TASK/'environment/model.py')==functions(BASE/'scripts/deforming_crystal_current_baseline.py')
    standard=BASE.parents[1]/'archives/deforming-crystal-current-r1/tasks/deforming-crystal-current'
    standard_files=['instruction.md','task.toml','environment/Dockerfile','environment/test_public.py','solution/solve.sh','tests/test.sh']
    for f in standard_files: assert (TASK/f).read_bytes()==(standard/f).read_bytes()
    report['input_audit']={'neutral_standard_byte_equal':standard_files,'starter_default_is_None':True,
        'completed_source_forward_AST_equal':True,'public_private_data_equal':True,
        'image_copy':'README.md model.py test_public.py data/ only',
        'true_parameter_private':'reference.py and metadata.json are outside environment/',
        'same_task_limits':{'agent':600,'verifier':60},
        'arbitrary_implementation_changes_permitted':True}
    report['local_controls']=local_controls()
    report['summary']={
        'domain_cases':len(report['domain_rows']),
        'max_domain_reference_error':max(r['error'] for r in report['domain_rows']),
        'graded_distinct_cases':len(report['graded_reference_rows']),
        'max_graded_reference_error':max(r['error'] for r in report['graded_reference_rows']),
        'max_graded_finer_change':max(r['finer_change'] for r in report['graded_reference_rows']),
        'noise_fits_per_control':len(report['noise_rows']),
        'max_noise_chi2':max(row[c]['chi2'] for row in report['noise_rows'] for c in ['oracle','shortcut']),
        'max_noise_parameter_error':max(row[c]['parameter_error'] for row in report['noise_rows'] for c in ['oracle','shortcut']),
        'max_noise_oracle_error':max(max(r['oracle']['errors'].values()) for r in report['noise_rows']),
        'min_noise_source_gap':min(v for r in report['noise_rows'] for g,v in r['shortcut']['errors'].items() if g!='short_anchor'),
        'min_sampled_parameter_source_gap':min(v['source_error'] for r in report['parameter_rows'] for g,v in r['groups'].items() if g!='short_anchor'),
        'all_expected_noise_outcomes':True}
    report['source_hashes']={str(p.relative_to(BASE)):digest(p) for p in [TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',TASK/'tests/test_hidden.py',BASE/'scripts/deforming_crystal_current_baseline.py',Path(__file__)]}
    report['status']='science_and_local_controls_complete'
    report['runtime_seconds']=time.perf_counter()-start


if __name__=='__main__':
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run=RESULTS/'executions'/stamp;run.mkdir(parents=True)
    shutil.copy2(__file__,run/'validator.py')
    report={'status':'running','run':str(run.relative_to(BASE)),'model_or_docker_runs':0}
    try:
        validate(report)
    except Exception:
        report['status']='author_validation_failed';report['traceback']=traceback.format_exc()
        (run/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        raise
    else:
        content=json.dumps(report,indent=2)+'\n'
        (run/'report.json').write_text(content)
        (RESULTS/'deforming-crystal-current-r2-validation.json').write_text(content)
        print(json.dumps({'status':report['status'],'summary':report['summary'],'runtime_seconds':report['runtime_seconds']},indent=2))
