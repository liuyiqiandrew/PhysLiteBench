"""Author validation and local completed controls; no model or Docker calls."""
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
from scipy.linalg import solve_continuous_lyapunov

BASE=Path(__file__).resolve().parents[1]
TASK=BASE/'tasks/coherent-array';RESULTS=BASE/'results';REPORT={}

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

oracle=load('oracle',TASK/'solution/model.py')
source=load('source',BASE/'scripts/coherent_array_baseline.py')
reference=load('reference',TASK/'tests/reference.py')
metadata=json.loads((TASK/'tests/metadata.json').read_text())

def error(x,y):return float(np.linalg.norm(x-y)/np.linalg.norm(y))

def controls():
    result={}
    for name,path in [('oracle',TASK/'solution/model.py'),('shortcut',BASE/'scripts/coherent_array_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='coherent-array-') as tmp:
            p=Path(tmp);shutil.copytree(TASK/'environment',p/'app');shutil.copytree(TASK/'tests',p/'tests')
            shutil.copyfile(path,p/'app/model.py')
            env=os.environ.copy();env.update(PYTHONPATH=str(p/'app'),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            start=time.monotonic()
            r=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',str(p/'app/test_public.py'),str(p/'tests/test_hidden.py')],cwd=p,env=env,capture_output=True,text=True,timeout=60)
            result[name]={'returncode':r.returncode,'seconds':time.monotonic()-start,'stdout':r.stdout,'stderr':r.stderr}
    (RESULTS/'coherent-array-r1-local-controls.json').write_text(json.dumps(result,indent=2)+'\n')
    assert result['oracle']['returncode']==0 and '7 passed' in result['oracle']['stdout']
    assert result['shortcut']['returncode']==1 and '3 failed, 4 passed' in result['shortcut']['stdout']
    return {k:{a:b for a,b in v.items() if a not in ['stdout','stderr']} for k,v in result.items()}

def run(generate):
    start=time.monotonic();k=reference.TRUE_PARAMETER;sigma=metadata['measurement_sigma']
    inputs=reference.calibration_inputs()*metadata['calibration_repeats'];clean=reference.predict(inputs,k)
    if generate:
        assert not (TASK/'environment/data/calibration.json').exists(),'Do not silently replace a dataset.'
        noisy=clean+np.random.default_rng(metadata['calibration_seed']).normal(0,sigma,len(clean))
        rows=[{'input':e,'value':float(v),'sigma':sigma} for e,v in zip(inputs,noisy)]
        text=json.dumps(rows,indent=2)+'\n'
        for side in ['environment','tests']:(TASK/side/'data/calibration.json').write_text(text)
    rows=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert len(rows)==288 and [r['input'] for r in rows]==inputs
    assert all(set(r)=={'input','value','sigma'} and r['sigma']==sigma for r in rows)
    assert (TASK/'environment/data/calibration.json').read_bytes()==(TASK/'tests/data/calibration.json').read_bytes()
    REPORT.update(status='validation_in_progress',model_evaluations=0,calibration_count=len(rows),calibration_distinct_settings=9,
                  true_stiffness=k,sigma=sigma,prediction_limit=.04)
    groups=reference.hidden_inputs();truths={g:reference.predict(v,k) for g,v in groups.items()}
    actual={}
    for name,module in [('oracle',oracle),('shortcut',source)]:
        m=module.Model().fit(rows);res=(m.predict(inputs)-[r['value'] for r in rows])/sigma
        actual[name]={'parameter':m.stiffness,'parameter_relative_error':abs(m.stiffness/k-1),'chi2':float(res@res/(len(rows)-1)),
                      'hidden':{g:error(m.predict(v),truths[g]) for g,v in groups.items()}}
    REPORT['actual_data']=actual
    REPORT['scored_reference']={'max_error':max(float(np.max(abs(oracle.predict_at(v,k)-truths[g]))) for g,v in groups.items()),
     'max_refinement':max(float(np.max(abs(reference.predict(v,k,48)-truths[g]))) for g,v in groups.items()),
     'minimum_diagnostic_signal':float(min(np.min(abs(truths[g])) for g in groups if g!='exact_anchor')),
     'calibration_reference_error_sigma':float(np.max(abs(oracle.predict_at(inputs,k)-clean))/sigma)}
    max_chi=max_param=max_oracle=max_anchor=0.;min_source=float('inf');fit_values=[]
    rng=np.random.default_rng(metadata['noise_seed'])
    for _ in range(256):
        noisy=clean+rng.normal(0,sigma,len(clean));sample=[{'input':e,'value':float(v),'sigma':sigma} for e,v in zip(inputs,noisy)]
        for name,module in [('oracle',oracle),('shortcut',source)]:
            m=module.Model().fit(sample);fit_values.append(m.stiffness)
            chi=float(np.sum(((m.predict(inputs)-noisy)/sigma)**2)/(len(inputs)-1));pe=abs(m.stiffness/k-1)
            max_chi=max(max_chi,chi);max_param=max(max_param,pe);assert chi<1.5 and pe<.03
            for g,v in groups.items():
                err=error(m.predict(v),truths[g])
                if name=='oracle':max_oracle=max(max_oracle,err);assert err<.04
                elif g=='exact_anchor':max_anchor=max(max_anchor,err);assert err<.04
                else:min_source=min(min_source,err);assert err>.04
    REPORT['noise']={'trials':256,'all_expected_outcomes':True,'max_chi2':max_chi,'max_parameter_error':max_param,
      'max_oracle_error':max_oracle,'max_shortcut_anchor_error':max_anchor,'min_shortcut_error':min_source,
      'fit_min':min(fit_values),'fit_max':max(fit_values)}
    max_fit=max_cal=0.;min_slope=float('inf');gaps={g:float('inf') for g in groups if g!='exact_anchor'}
    for stiffness in np.linspace(.8,1.2,41):
        y=oracle.predict_at(inputs,stiffness);max_cal=max(max_cal,float(np.max(abs(source.predict_at(inputs,stiffness)-y))))
        sample=[{'input':e,'value':float(v),'sigma':sigma} for e,v in zip(inputs,y)]
        for module in [oracle,source]:max_fit=max(max_fit,abs(module.Model().fit(sample).stiffness-stiffness))
        eigen,vectors=oracle.modes(1.);inv=(vectors/eigen)@vectors.T
        dist=np.diag(inv)[:,None]+np.diag(inv)[None,:]-2*inv
        for t in [.04,.08,.12]:
            a=(2*np.pi)**2*t*dist/2;min_slope=min(min_slope,float(np.sum(a*np.exp(-a/stiffness))/(5*stiffness**2)))
        for g,v in groups.items():
            if g!='exact_anchor':gaps[g]=min(gaps[g],error(source.predict_at(v,stiffness),oracle.predict_at(v,stiffness)))
    assert max_fit<1e-7 and max_cal<1e-12 and min(gaps.values())>.04
    REPORT['identifiability']={'global_proof':'At reciprocal q, all equal-time pair phases are+1 and S=N^-1 sum exp[-a_ij/K] with positive off-diagonal a_ij. Hence dS/dK>0, so these static records alone identify K globally.',
      'parameter_samples':41,'max_noiseless_recovery':max_fit,'max_calibration_equivalence':max_cal,
      'sampled_minimum_static_derivative':min_slope,'minimum_group_gaps':gaps}
    cases=list(product([.8,1.,1.2],[.04,.08,.12],[0.,.25,.5,.75,1.],[1.,2.5,4.,2*np.pi],[0.,.5,1.5,4.]))
    rng=np.random.default_rng(171127)
    cases.extend((rng.uniform(.8,1.2),rng.uniform(.04,.12),rng.uniform(0,1),rng.uniform(1,2*np.pi),rng.uniform(0,4)) for _ in range(64))
    max_ref=max_refine=max_qsym=max_equal=max_uncoupled=0.;min_variance=float('inf')
    for stiffness,t,c,q,d in cases:
        exact=oracle.response(stiffness,t,c,q,d);ref=reference.response(stiffness,t,c,q,d);fine=reference.response(stiffness,t,c,q,d,48)
        max_ref=max(max_ref,abs(exact-ref));max_refine=max(max_refine,abs(ref-fine))
        max_qsym=max(max_qsym,abs(exact-oracle.response(stiffness,t,c,-q,d)))
        max_equal=max(max_equal,abs(oracle.response(stiffness,t,c,q,0)-source.response(stiffness,t,c,q,0)))
        max_uncoupled=max(max_uncoupled,abs(oracle.response(stiffness,t,0.,q,d)-source.response(stiffness,t,0.,q,d)))
        ev,v=oracle.modes(c);diag=np.diag((v*(t/(stiffness*ev)))@v.T)
        plateau=abs(np.sum(np.exp(1j*q*np.arange(5)-q*q*diag/2)))**2/5
        min_variance=min(min_variance,oracle.response(stiffness,t,c,q,0)-plateau)
    assert max(max_ref,max_refine)<1e-8 and max(max_qsym,max_equal,max_uncoupled)<1e-12 and min_variance>0
    REPORT['full_domain']={'structured_cases':720,'random_cases':64,'max_reference_error':max_ref,'max_reference_refinement':max_refine,
      'max_wavevector_reflection_error':max_qsym,'max_equal_time_equivalence':max_equal,'max_uncoupled_equivalence':max_uncoupled,'minimum_connected_variance':min_variance,
      'scope':'Bounded grids include coupling interiors and random full-domain controls; not an exhaustive interval error bound.'}
    min_eig=float('inf');max_lyap=max_energy=max_balance=max_plateau=0.
    for stiffness,t,c,q in product([.8,1.2],[.04,.12],[0.,.4,1.],[1.,4.,2*np.pi]):
        times=np.linspace(0,4,17)
        for module in [oracle,source]:
            kernel=np.array([[module.response(stiffness,t,c,q,abs(x-y)) for x in times] for y in times])
            min_eig=min(min_eig,float(np.linalg.eigvalsh(kernel).min()))
        ev,v=oracle.modes(c);hessian=(v*(stiffness*ev))@v.T
        drift=np.block([[np.zeros((5,5)),np.eye(5)],[-hessian,-.4*np.eye(5)]])
        noise=np.zeros((10,10));noise[5:,5:]=.8*t*np.eye(5)
        stationary=solve_continuous_lyapunov(drift,-noise)
        max_lyap=max(max_lyap,float(np.max(abs(drift@stationary+stationary@drift.T+noise))))
        energy=.5*np.trace(stationary[5:,5:])+.5*np.trace(hessian@stationary[:5,:5])
        max_energy=max(max_energy,abs(energy-5*t));max_balance=max(max_balance,abs(.4*np.trace(stationary[5:,5:])-2*t))
        diag=np.diag(stationary[:5,:5]);plateau=abs(np.sum(np.exp(1j*q*np.arange(5)-q*q*diag/2)))**2/5
        for module in [oracle,source]:max_plateau=max(max_plateau,abs(module.response(stiffness,t,c,q,200.)-plateau))
    assert min_eig>-1e-10 and max(max_lyap,max_energy,max_balance,max_plateau)<1e-10
    REPORT['limits']={'covariance_kernel_samples':36,'time_points':17,'minimum_kernel_eigenvalue':min_eig,
      'max_stationary_lyapunov_residual':max_lyap,'max_total_energy_equipartition_error':max_energy,
      'max_bath_injection_dissipation_balance_error':max_balance,'max_long_time_plateau_error':max_plateau,
      'positive_kernel_proof':'The source is a positive multiple of the sum of exact connected self-field covariance kernels plus the nonnegative exact mean-field constant. It is a valid alternative stationary correlation kernel.',
      'qualification':'Baths exchange energy and momentum. No isolated total-energy or total-momentum conservation is asserted; the equilibrium balance is checked.'}
    def funcs(path):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef)}
    assert funcs(TASK/'environment/model.py')==funcs(BASE/'scripts/coherent_array_baseline.py')
    REPORT['local_controls']=controls();REPORT['status']='science_and_local_controls_complete';REPORT['seconds']=time.monotonic()-start
    REPORT['source_hashes']={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',TASK/'tests/test_hidden.py',BASE/'scripts/coherent_array_baseline.py',Path(__file__)]}
    (RESULTS/'coherent-array-r1-validation.json').write_text(json.dumps(REPORT,indent=2)+'\n')
    print(json.dumps(REPORT,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true')
    try:run(parser.parse_args().generate)
    except Exception:
        REPORT['status']='author_check_failed';REPORT['traceback']=traceback.format_exc()
        (RESULTS/f'author-failure-{time.time_ns()}.json').write_text(json.dumps(REPORT,indent=2)+'\n')
        raise
