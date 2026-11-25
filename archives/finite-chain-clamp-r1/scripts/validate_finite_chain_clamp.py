"""Author science and isolated local controls; no model or Docker launches."""
import argparse
import ast
from itertools import product
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
from scipy.integrate import quad

BASE = Path(__file__).resolve().parents[1]
TASK = BASE/'tasks/finite-chain-clamp'
RESULTS = BASE/'results'
REPORT = {}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle = load('chain_oracle', TASK/'solution/model.py')
shortcut = load('chain_shortcut', BASE/'scripts/finite_chain_clamp_baseline.py')
reference = load('chain_reference', TASK/'tests/reference.py')
metadata = json.loads((TASK/'tests/metadata.json').read_text())


def nrmse(value, truth):
    return float(np.sqrt(np.mean((value-truth)**2)/np.mean(truth**2)))


def density(n, b, x):
    s = (n-abs(x)/b)/2
    if s <= 0:
        return 0.
    return math.fsum((-1)**j*math.comb(n,j)*(s-j)**(n-1)
                     for j in range(min(n,int(s))+1))/math.factorial(n-1)/(2*b)


def local_controls():
    controls = {}
    for name, path in [('oracle', TASK/'solution/model.py'),
                       ('shortcut', BASE/'scripts/finite_chain_clamp_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='finite-chain-clamp-') as folder:
            location = Path(folder)
            shutil.copytree(TASK/'environment', location/'app')
            shutil.copytree(TASK/'tests', location/'tests')
            shutil.copy2(path, location/'app/model.py')
            env = os.environ.copy()
            env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(location/'app'),
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
            start = time.monotonic()
            result = subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
                                     str(location/'app/test_public.py'),str(location/'tests/test_hidden.py')],
                                    cwd=location,env=env,capture_output=True,text=True,timeout=60)
            controls[name] = {'returncode':result.returncode,'seconds':time.monotonic()-start,
                              'stdout':result.stdout,'stderr':result.stderr}
    (RESULTS/'finite-chain-clamp-r1-local-controls.json').write_text(json.dumps(controls,indent=2)+'\n')
    assert controls['oracle']['returncode'] == 0 and '7 passed' in controls['oracle']['stdout']
    assert controls['shortcut']['returncode'] == 1 and '3 failed, 4 passed' in controls['shortcut']['stdout']
    return {name:{k:v for k,v in row.items() if k not in ('stdout','stderr')}
            for name,row in controls.items()}


def run(generate=False):
    start = time.monotonic()
    true = reference.TRUE_PARAMETER
    inputs = reference.calibration_inputs()*metadata['calibration_repeats']
    sigma = metadata['measurement_sigma']
    clean = reference.predict(inputs,true)
    if generate:
        values = clean+np.random.default_rng(metadata['calibration_seed']).normal(0,sigma,len(inputs))
        rows = [{'input':e,'value':float(v),'sigma':sigma} for e,v in zip(inputs,values)]
        text = json.dumps(rows,indent=2)+'\n'
        for side in ('environment','tests'):
            (TASK/side/'data/calibration.json').write_text(text)
    rows = json.loads((TASK/'environment/data/calibration.json').read_text())
    assert [r['input'] for r in rows] == inputs and len(rows) == 324
    assert all(r['sigma'] == sigma for r in rows)
    assert (TASK/'environment/data/calibration.json').read_bytes() == (TASK/'tests/data/calibration.json').read_bytes()
    REPORT.update(revision=1,status='science_in_progress',model_evaluations=0,
                  true_link_length=true,calibration_seed=metadata['calibration_seed'],
                  noise_seed=metadata['noise_seed'],calibration_count=len(rows),
                  distinct_settings=len(reference.calibration_inputs()),sigma=sigma,
                  prediction_limit=metadata['prediction_limit'])
    groups = reference.hidden_inputs()
    truths = {k:reference.predict(v,true) for k,v in groups.items()}
    actual = {}
    for name,mod in [('oracle',oracle),('shortcut',shortcut)]:
        model = mod.Model().fit(rows)
        residual = (model.predict(inputs)-np.array([r['value'] for r in rows]))/sigma
        actual[name] = {'link_length':model.link_length,
                        'parameter_relative_error':abs(model.link_length/true-1),
                        'calibration_chi2':float(residual@residual/(len(rows)-1)),
                        'hidden':{k:nrmse(model.predict(v),truths[k]) for k,v in groups.items()}}
        assert actual[name]['parameter_relative_error']<.03 and actual[name]['calibration_chi2']<1.5
    REPORT['actual_data'] = actual
    max_hidden_ref = max(float(np.max(abs(oracle.predict_at(v,true)-truths[k]))) for k,v in groups.items())
    max_hidden_refine = max(float(np.max(abs(reference.predict(v,true,512,48)-truths[k]))) for k,v in groups.items())
    cal_bias = float(np.max(abs(oracle.predict_at(inputs,true)-clean))/sigma)
    REPORT['scored_reference'] = {'max_absolute_error':max_hidden_ref,'max_refinement_change':max_hidden_refine,
                                  'calibration_bias_in_sigma':cal_bias,
                                  'minimum_clamp_signal':float(min(truths[k].min() for k in groups if k!='force_anchor'))}
    assert max(max_hidden_ref,max_hidden_refine)<1e-6 and cal_bias<.01
    # The supplied forward code and completed shortcut are identical.
    def functions(path):
        return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(path.read_text()).body
                if isinstance(node,ast.FunctionDef)}
    assert functions(TASK/'environment/model.py') == functions(BASE/'scripts/finite_chain_clamp_baseline.py')
    rng = np.random.default_rng(metadata['noise_seed'])
    max_chi = max_parameter = max_oracle = max_anchor = 0.
    min_shortcut = float('inf'); fits=[]
    for _ in range(metadata['noise_trials']):
        noisy = clean+rng.normal(0,sigma,len(clean))
        records = [{'input':e,'value':float(v),'sigma':sigma} for e,v in zip(inputs,noisy)]
        for name,mod in [('oracle',oracle),('shortcut',shortcut)]:
            model = mod.Model().fit(records); fits.append(model.link_length)
            chi = float(np.sum(((model.predict(inputs)-noisy)/sigma)**2)/(len(inputs)-1))
            parameter = abs(model.link_length/true-1)
            max_chi=max(max_chi,chi);max_parameter=max(max_parameter,parameter)
            assert chi<1.5 and parameter<.03
            for k,v in groups.items():
                error=nrmse(model.predict(v),truths[k])
                if name=='oracle':max_oracle=max(max_oracle,error);assert error<.04
                elif k=='force_anchor':max_anchor=max(max_anchor,error);assert error<.04
                else:min_shortcut=min(min_shortcut,error);assert error>.04
    REPORT['noise']={'trials':metadata['noise_trials'],'all_expected_outcomes':True,
                     'fit_min':min(fits),'fit_max':max(fits),'max_calibration_chi2':max_chi,
                     'max_parameter_relative_error':max_parameter,'max_oracle_hidden_error':max_oracle,
                     'min_shortcut_diagnostic_error':min_shortcut,'max_shortcut_anchor_error':max_anchor}
    # Strict positive derivative proves global uniqueness; scan endpoints too.
    max_recovery=max_cal_difference=max_cal_reference=max_fd=0.
    min_slope=float('inf');min_gap=float('inf');gap_by_group={k:float('inf') for k in groups if k!='force_anchor'}
    for b in np.linspace(.8,1.2,41):
        y=oracle.predict_at(inputs,b)
        max_cal_difference=max(max_cal_difference,float(np.max(abs(y-shortcut.predict_at(inputs,b)))))
        max_cal_reference=max(max_cal_reference,float(np.max(abs(y-reference.predict(inputs,b)))))
        records=[{'input':e,'value':float(v),'sigma':sigma} for e,v in zip(inputs,y)]
        for mod in (oracle,shortcut):max_recovery=max(max_recovery,abs(mod.Model().fit(records).link_length-b))
        for e in reference.calibration_inputs():
            z=e['force']*b/e['temperature']
            slope=e['links']*(1/np.tanh(z)-z/np.sinh(z)**2)
            fd=(oracle.predict_at([e],b+1e-5)[0]-oracle.predict_at([e],b-1e-5)[0])/(2e-5)
            min_slope=min(min_slope,slope);max_fd=max(max_fd,abs(fd-slope))
        for k,v in groups.items():
            if k!='force_anchor':
                gap=nrmse(shortcut.predict_at(v,b),oracle.predict_at(v,b))
                min_gap=min(min_gap,gap);gap_by_group[k]=min(gap_by_group[k],gap)
    REPORT['identifiability']={'proof':'For z=F*b/T>0, d mean_x/db=N*(coth(z)-z*csch(z)^2)>0 because sinh(z)*cosh(z)>z. Every individual nonzero-force setting identifies b globally.',
                               'parameter_samples':41,'max_noiseless_fit_error':max_recovery,
                               'minimum_positive_derivative':min_slope,'max_derivative_finite_difference_error':max_fd,
                               'max_shared_calibration_error':max_cal_difference,'max_calibration_reference_error':max_cal_reference,
                               'min_diagnostic_gap_over_parameter':min_gap,'group_minima':gap_by_group}
    assert max_recovery<1e-7 and min_slope>0 and max_fd<1e-7 and max_cal_difference<1e-12
    assert max_cal_reference<1e-10 and min_gap>.04
    # Every allowed N, full parameter interval, boundary controls, and random interiors.
    cases=list(product(range(4,13),np.linspace(.8,1.2,11),[.6,1.4],[.15,.375,.6]))
    random=np.random.default_rng(161063)
    cases += [(int(random.integers(4,13)),random.uniform(.8,1.2),random.uniform(.6,1.4),random.uniform(.15,.6))
              for _ in range(72)]
    max_ref=max_refine=max_odd=max_scale=0.;min_force=min_density=min_case_gap=float('inf')
    gap_by_n={n:float('inf') for n in range(4,13)}
    for n,b,t,xper in cases:
        x=n*xper;physical=oracle.holding_force(n,b,t,x);source=shortcut.holding_force(n,b,t,x)
        ref=reference.holding_force(n,b,t,x);fine=reference.holding_force(n,b,t,x,512,48)
        max_ref=max(max_ref,abs(ref-physical));max_refine=max(max_refine,abs(ref-fine))
        max_odd=max(max_odd,abs(oracle.holding_force(n,b,t,-x)+physical))
        max_scale=max(max_scale,abs(oracle.holding_force(n,1.3*b,t,1.3*x)*1.3-physical),
                      abs(oracle.holding_force(n,b,1.2*t,x)-1.2*physical))
        min_force=min(min_force,physical);min_density=min(min_density,density(n,b,x))
        gap=abs(source-physical)/physical;min_case_gap=min(min_case_gap,gap);gap_by_n[n]=min(gap_by_n[n],gap)
        assert physical>0 and source>0 and np.isfinite(physical+source)
    REPORT['full_domain']={'cases':len(cases),'links':list(range(4,13)),
                           'max_fourier_reference_error':max_ref,'max_fourier_refinement':max_refine,
                           'max_odd_symmetry_error':max_odd,'max_unit_scaling_error':max_scale,
                           'minimum_physical_force':min_force,'minimum_projected_density':min_density,
                           'minimum_pointwise_source_gap':min_case_gap,'pointwise_gap_minima_by_links':gap_by_n,
                           'qualification':'The minimum N12 gap is close to the .04 gate; scored groups use N4,N6,N8. Full public-domain checks are retained without claiming all arbitrary groups fail.'}
    assert max(max_ref,max_refine)<2e-6 and max(max_odd,max_scale)<1e-9
    assert min_force>0 and min_density>0
    density_checks=[]
    for n in range(4,13):
        knots=np.arange(-n,n+1,2,dtype=float)
        norm=quad(lambda x:density(n,1.,x),-n,n,points=knots[1:-1],epsabs=1e-11)[0]
        variance=quad(lambda x:x*x*density(n,1.,x),-n,n,points=knots[1:-1],epsabs=1e-11)[0]
        density_checks.append({'links':n,'normalization_error':abs(norm-1),'variance_error':abs(variance-n/3)})
    support_error=max(abs(oracle.holding_force(n,1.,1.,n-.4)-(n-1)/.4) for n in range(4,13))
    soft=[]
    for n,b,t,p in [(4,.8,1.4,.6),(4,1.2,.6,.15),(8,1.,1.,.4),(12,.8,.6,.6)]:
        x=n*p;exact=oracle.holding_force(n,b,t,x)
        values=[reference.holding_force(n,b,t,x,256,40,k) for k in [100,200,400,800]]
        coarse=(values[0]-6*values[1]+8*values[2])/3
        fine=(values[1]-6*values[2]+8*values[3])/3
        soft.append({'links':n,'length':b,'temperature':t,'extension':x,'physical':exact,
                     'stiffnesses':[100,200,400,800],'mechanical_clamp_forces':values,
                     'extrapolation_error':abs(coarse-exact),'refined_extrapolation_error':abs(fine-exact)})
        assert abs(fine-exact)<max(2e-5,abs(coarse-exact))
    large=[]
    for n in [4,8,16,32,64,128]:
        f=reference.holding_force(n,1.,1.,.2*n,64,64);s=shortcut.holding_force(n,1.,1.,.2*n)
        large.append({'links':n,'physical':f,'source':s,'relative_gap':abs(s/f-1)})
    assert all(large[j+1]['relative_gap']<large[j]['relative_gap'] for j in range(len(large)-1))
    REPORT['limits']={'density_checks':density_checks,'support_interval_error':support_error,
                      'finite_clamp':soft,'large_chain':large,
                      'one_link':'For a single fixed-length freely oriented link, the axial density is constant in its interior and holding force is zero; this is an author-only limit, not a scored input.',
                      'scope':'Sign/scaling, support endpoint, finite-K and large-N checks can extend beyond the scored control domain. The finite-K reference is a mechanical interpretation check; grading uses infinite-K Fourier inversion.'}
    assert max(r['normalization_error'] for r in density_checks)<1e-10
    assert max(r['variance_error'] for r in density_checks)<1e-10 and support_error<1e-10
    REPORT['local_controls']=local_controls()
    REPORT['source_hashes']={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',
                                      BASE/'scripts/finite_chain_clamp_baseline.py',Path(__file__)]}
    REPORT['status']='science_and_local_controls_complete';REPORT['seconds']=time.monotonic()-start
    (RESULTS/'finite-chain-clamp-r1-validation.json').write_text(json.dumps(REPORT,indent=2)+'\n')
    print(json.dumps({k:REPORT[k] for k in ('status','actual_data','noise','identifiability','full_domain','local_controls','seconds')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true')
    try:
        run(parser.parse_args().generate)
    except Exception:
        REPORT['status']='author_check_failed';REPORT['traceback']=traceback.format_exc()
        (RESULTS/f'author-check-failure-{time.time_ns()}.json').write_text(json.dumps(REPORT,indent=2)+'\n')
        raise
