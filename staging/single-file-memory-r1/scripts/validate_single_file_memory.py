"""Scientific checks and isolated local controls; never launches agent evaluations."""
import argparse
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np

STAGE = Path(__file__).resolve().parents[1]
TASK = STAGE/'tasks/single-file-memory'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle = load('physical_oracle', TASK/'solution/model.py')
shortcut = load('completed_source', STAGE/'scripts/single_file_memory_baseline.py')
reference = load('crossing_reference', TASK/'tests/reference.py')
SIGMA = .002
SEED = 119231
NOISE_SEED = 119233
LIMIT = .04


def records(values, experiments):
    return [dict(input=e, value=float(y), sigma=SIGMA) for e,y in zip(experiments,values)]


def fit_metrics(module, data, truth):
    model = module.Model().fit(data)
    predictions = model.predict([r['input'] for r in data])
    error = (predictions-np.array([r['value'] for r in data]))/SIGMA
    hidden = {name:float(np.linalg.norm(model.predict(experiments)-truth[name])/np.linalg.norm(truth[name]))
              for name,experiments in reference.hidden_inputs().items()}
    return dict(parameter=model.diffusivity, parameter_relative_error=abs(model.diffusivity/reference.TRUE_PARAMETER-1),
                calibration_chi2=float(error@error)/(len(data)-1), hidden=hidden)


def local_controls():
    results = {}
    for name,source in [('oracle',TASK/'solution/model.py'),('shortcut',STAGE/'scripts/single_file_memory_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='single-file-control-') as directory:
            directory=Path(directory)
            shutil.copytree(TASK/'environment', directory/'app')
            shutil.copytree(TASK/'tests', directory/'tests')
            shutil.copy2(source, directory/'app/model.py')
            env=dict(os.environ, PYTHONPATH=str(directory/'app'), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
            start=time.monotonic()
            run=subprocess.run([sys.executable,'-m','pytest','-q',str(directory/'app/test_public.py'),str(directory/'tests/test_hidden.py')],cwd=directory,env=env,capture_output=True,text=True)
            results[name]={'returncode':run.returncode,'seconds':time.monotonic()-start,'stdout':run.stdout,'stderr':run.stderr}
    assert results['oracle']['returncode']==0 and '7 passed' in results['oracle']['stdout']
    assert results['shortcut']['returncode']==1 and '3 failed, 4 passed' in results['shortcut']['stdout']
    for name in ['ratio_two','ratio_four','ratio_eight']:
        assert f'test_prediction[{name}]' in results['shortcut']['stdout']
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');args=parser.parse_args()
    experiments=reference.calibration_inputs()
    noiseless=oracle.predict_at(experiments, reference.TRUE_PARAMETER)
    if args.generate:
        values=noiseless+np.random.default_rng(SEED).normal(0,SIGMA,len(experiments))
        data=records(values,experiments)
        for path in [TASK/'environment/data/calibration.json', TASK/'tests/data/calibration.json']:
            path.write_text(json.dumps(data,indent=2)+'\n')
        meta={'task':'single-file-memory','revision':1,'parameter':'diffusivity','true_parameter':reference.TRUE_PARAMETER,
              'prediction_limit':LIMIT,'calibration_chi2_limit':1.5,'parameter_relative_error_limit':.03,
              'sigma':SIGMA,'uncertainty_definition':'Fixed instrument uncertainty in L0 squared; independent of diffusivity and response.',
              'calibration_seed':SEED,'noise_validation_seed':NOISE_SEED,'calibration_records':len(data)}
        (TASK/'tests/metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    data=json.loads((TASK/'tests/data/calibration.json').read_text())
    assert data==json.loads((TASK/'environment/data/calibration.json').read_text())
    groups=reference.hidden_inputs()
    truth={name:reference.predict(es,reference.TRUE_PARAMETER) for name,es in groups.items()}
    actual={name:fit_metrics(module,data,truth) for name,module in [('oracle',oracle),('shortcut',shortcut)]}
    noise=[];rng=np.random.default_rng(NOISE_SEED)
    for _ in range(256):
        draw=records(noiseless+rng.normal(0,SIGMA,len(data)),experiments)
        noise.append({name:fit_metrics(module,draw,truth) for name,module in [('oracle',oracle),('shortcut',shortcut)]})
    for result in [actual]+noise:
        for name in ['oracle','shortcut']:
            assert result[name]['parameter_relative_error']<.03 and result[name]['calibration_chi2']<1.5
        assert max(result['oracle']['hidden'].values())<LIMIT
        assert result['shortcut']['hidden']['diagonal_anchors']<LIMIT
        assert min(result['shortcut']['hidden'][name] for name in groups if name!='diagonal_anchors')>LIMIT
    # Complete parameter interval: exact calibration equivalence and injective sqrt(D) amplitude.
    recovery=[];equivalence=[]
    for d in np.linspace(.8,1.2,41):
        y=oracle.predict_at(experiments,d);equivalence.append(float(np.max(abs(y-shortcut.predict_at(experiments,d)))))
        for model in [oracle,shortcut]:recovery.append(abs(model.Model().fit(records(y,experiments)).diffusivity-d))
    design=oracle.predict_at(experiments,1.)
    # Independent reference at hidden inputs, corners, near-equal times, and off-grid controls.
    points=[(d,rho,t,s) for d,rho,t,s in itertools.product([.8,1.2],[.7,1.5],[.25,4.],[.25,4.])]
    near=[]
    for t,d,rho in itertools.product([.25,.7,2.,4.],[.8,1.2],[.7,1.5]):
        for delta in [0.,1e-14,1e-10,1e-6,.001]:
            s=t/(1+delta) if t>.25 else t*(1+delta)
            near.append((d,rho,t,s))
    points+=near
    rng=np.random.default_rng(119239)
    points += [tuple([rng.uniform(.8,1.2),rng.uniform(.7,1.5),*rng.uniform(.25,4,2)]) for _ in range(32)]
    errors=[];refinement=[];symmetry=[];scaling=[];probability_bounds=[]
    for d,rho,t,s in points:
        ref=reference.covariance(d,rho,t,s)
        errors.append(abs(ref-oracle.covariance(d,rho,t,s)))
        refinement.append(abs(ref-reference.covariance(d,rho,t,s,768,192)))
        symmetry.append(abs(ref-reference.covariance(d,rho,s,t)))
        scaling.append(abs(oracle.covariance(d,rho,5*t,5*s)-np.sqrt(5)*oracle.covariance(d,rho,t,s)))
        probability_bounds.append(ref/np.sqrt(oracle.covariance(d,rho,t,t)*oracle.covariance(d,rho,s,s)))
    psd=[]
    for _ in range(48):
        times=np.sort(rng.uniform(.25,4,20));d=rng.uniform(.8,1.2);rho=rng.uniform(.7,1.5)
        psd.append([float(np.linalg.eigvalsh(np.array([[m.covariance(d,rho,t,s) for t in times] for s in times])).min()) for m in [oracle,shortcut]])
    assert max(errors)<3e-8 and max(refinement)<3e-8 and max(symmetry)<1e-12
    assert min(np.min(psd,axis=0))>0 and min(probability_bounds)>=0 and max(probability_bounds)<1+1e-12
    assert max(recovery)<1e-12 and max(equivalence)<1e-14
    # Exact same one-time variance does not fix increments taken after waiting.
    increment=lambda m,w,dt:m.covariance(1,1,w+dt,w+dt)+m.covariance(1,1,w,w)-2*m.covariance(1,1,w+dt,w)
    memory=[{'waiting':w,'oracle_increment_variance':increment(oracle,w,1),'shortcut_increment_variance':increment(shortcut,w,1)} for w in [0,.25,1,4,1000]]
    assert memory[-1]['oracle_increment_variance']>memory[0]['oracle_increment_variance']*1.4
    report={'status':'scientific_validation_passed_no_model_evaluations','calibration_records':len(data),'unique_calibration_settings':len({json.dumps(e,sort_keys=True) for e in experiments}),
            'calibration_seed':SEED,'noise_seed':NOISE_SEED,'sigma':SIGMA,'prediction_limit':LIMIT,'actual':actual,
            'noise_realizations':256,'all_calibration_and_parameter_checks_pass':True,'all_oracles_pass':True,'all_shortcuts_fail_three_groups':True,
            'max_noise_parameter_relative_error':max(x['oracle']['parameter_relative_error'] for x in noise),
            'max_noise_chi2':max(x['oracle']['calibration_chi2'] for x in noise),
            'max_noise_oracle_hidden_error':max(max(x['oracle']['hidden'].values()) for x in noise),
            'minimum_noise_shortcut_discriminating_error':min(x['shortcut']['hidden'][g] for x in noise for g in groups if g!='diagonal_anchors'),
            'full_interval_noiseless_fit_points':41,'noiseless_fit_max_absolute_error':max(recovery),'calibration_equivalence_max':max(equivalence),
            'analytic_identifiability':'For each diagonal setting C=A*sqrt(D) with known A>0. Weighted least squares is strictly convex in sqrt(D), so the constrained noiseless fit is unique.',
            'minimum_calibration_derivative':float(np.min(design)/(2*np.sqrt(1.2))),
            'reference_points':len(points),'near_diagonal_points':len(near),'max_reference_error':max(errors),'max_reference_refinement':max(refinement),'max_time_exchange_error':max(symmetry),'joint_time_scaling_error':max(scaling),
            'minimum_covariance_eigenvalues':np.min(psd,axis=0).tolist(),'correlation_range':[min(probability_bounds),max(probability_bounds)],'waiting_increment_check':memory,
            'prototype_evidence':'prototype/assessment.json; independent finite-rank simulations are supplementary only and were not rerun.',
            'all_noise_metrics':noise}
    (STAGE/'results/single-file-memory-r1-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    local=local_controls();(STAGE/'results/single-file-memory-r1-local-controls.json').write_text(json.dumps(local,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='all_noise_metrics'},indent=2))
    print(json.dumps({k:{'returncode':v['returncode'],'seconds':v['seconds']} for k,v in local.items()},indent=2))

if __name__=='__main__':main()
