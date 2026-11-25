"""Scientific checks; generation is explicit. Does not launch model evaluations."""
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

import numpy as np
from scipy.linalg import eigvals

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/magnetic-bath-transfer'

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

oracle = load('oracle', TASK/'solution/model.py')
shortcut = load('shortcut', ROOT/'scripts/magnetic_bath_transfer_baseline.py')
reference = load('reference', TASK/'tests/reference.py')

def errors(model, groups, truth):
    return {key: float(np.sqrt(np.mean((model.predict(inputs)-truth[key])**2)) /
                       np.sqrt(np.mean(truth[key]**2))) for key,inputs in groups.items()}

def local_control(source, label):
    target = ROOT/'results/local-controls'/label
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='magnetic-bath-control-') as temp:
        app = Path(temp)
        shutil.copy2(source, app/'model.py')
        shutil.copy2(TASK/'environment/test_public.py', app/'test_public.py')
        shutil.copytree(TASK/'environment/data', app/'data')
        env = dict(os.environ, PYTHONPATH=str(app)+os.pathsep+str(TASK/'tests'),
                   MODEL_PATH=str(app/'model.py'), METRICS_PATH=str(target/'metrics.json'),
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        run = subprocess.run([sys.executable,'-m','pytest','-q',str(app/'test_public.py'),
                              str(TASK/'tests/test_hidden.py')], env=env,
                             text=True, capture_output=True, timeout=60)
        (target/'stdout.txt').write_text(run.stdout+run.stderr)
    return {'exit_code': run.returncode, 'metrics':json.loads((target/'metrics.json').read_text()),
            'stdout':run.stdout+run.stderr}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true')
    args=parser.parse_args()
    inputs=reference.calibration_inputs(); exact=reference.predict(inputs)
    sigma=.0006; seed=78113
    if args.generate:
        measured=exact+np.random.default_rng(seed).normal(0,sigma,len(inputs))
        records=[{'input':e,'value':float(y),'sigma':sigma} for e,y in zip(inputs,measured)]
        data=json.dumps(records,indent=2)+'\n'
        for directory in ['environment','tests']:
            (TASK/directory/'data/calibration.json').write_text(data)
    records=json.loads((TASK/'tests/data/calibration.json').read_text())
    assert (TASK/'tests/data/calibration.json').read_bytes()==(TASK/'environment/data/calibration.json').read_bytes()
    assert hashlib.sha256((TASK/'instruction.md').read_bytes()).hexdigest()=='6a5ef18a542d5fd42298ccab151355ecf224974f3b465d9a6299768988739b62'
    groups=reference.hidden_inputs();truth={k:reference.predict(v) for k,v in groups.items()}
    nominal={}
    for key,module in [('oracle',oracle),('shortcut',shortcut)]:
        model=module.Model().fit(records)
        residual=(model.predict(inputs)-np.array([r['value'] for r in records]))/sigma
        nominal[key]={'friction':model.friction,'chi2':float(np.sum(residual**2)/(len(inputs)-1)),
                      'hidden':errors(model,groups,truth)}
    exact_records=[dict(r,value=float(v)) for r,v in zip(records,exact)]
    calibration_recovery=[]
    for gamma in np.linspace(.7,1.5,41):
        ys=reference.predict(inputs,float(gamma))
        rec=[dict(r,value=float(v)) for r,v in zip(records,ys)]
        calibration_recovery.append(abs(shortcut.Model().fit(rec).friction-gamma))
    # The analytic slope is negative for every public k,h,m,gamma combination.
    monotonicity_margin=.8*.7**2-.8*.55**2
    assert monotonicity_margin>0
    grid=np.linspace(.7,1.5,401)
    loss=[]
    for gamma in grid:
        model=shortcut.Model();model.friction=float(gamma)
        loss.append(float(np.sum(((model.predict(inputs)-exact)/sigma)**2)))
    assert np.sum(np.diff(np.sign(np.diff(loss)))>0)==1
    rng=np.random.default_rng(78117);noise=[]
    for _ in range(256):
        values=exact+rng.normal(0,sigma,len(inputs))
        rec=[dict(r,value=float(v)) for r,v in zip(records,values)]
        models=[oracle.Model().fit(rec),shortcut.Model().fit(rec)]
        chi=[float(np.sum(((m.predict(inputs)-values)/sigma)**2)/(len(inputs)-1)) for m in models]
        errs=[errors(m,groups,truth) for m in models]
        noise.append(dict(friction=[m.friction for m in models],chi2=chi,hidden=errs))
        assert max(chi)<1.5
        assert max(abs(m.friction/reference.TRUE_PARAMETER-1) for m in models)<.03
        assert max(errs[0].values())<.04 and min(errs[1].values())>.04
    extrema=[]
    for offset in [-3,3]:
        rec=[dict(r,value=float(v+offset*sigma)) for r,v in zip(records,exact)]
        models=[oracle.Model().fit(rec),shortcut.Model().fit(rec)]
        errs=[errors(m,groups,truth) for m in models]
        extrema.append(dict(offset_sigma=offset,friction=[m.friction for m in models],hidden=errs))
        assert max(errs[0].values())<.04 and min(errs[1].values())>.04
    independent=0.; closed_error=0.; energy_error=0.; parity_error=0.; equal_temp=0.
    min_entropy=np.inf; min_covariance=np.inf; max_drift_real=-np.inf; cal_difference=0.
    rng=np.random.default_rng(78121)
    spots=[]
    for _ in range(96):
        gamma=rng.uniform(.7,1.5);m=rng.uniform(.15,.8);k=rng.uniform(.8,1.8)
        h=rng.uniform(.15,.55);B=rng.uniform(-1.2,1.2);T=rng.uniform(.5,2.,2)
        e=dict(mass=m,spring=k,coupling=h,field=B,temperatures=T.tolist(),bath=0)
        got=oracle.response(e,gamma);ref=reference.predict([e],gamma)[0]
        algebra=gamma*(T[0]-T[1])*(B*B+m*h*h/k)/(2*m*(gamma*gamma+B*B+m*h*h/k))
        base=shortcut.response(e,gamma)
        base_algebra=gamma*(T[0]-T[1])*h*h/(2*k*(gamma*gamma+B*B+m*h*h/k))
        C=oracle.stationary_covariance(e,gamma);K=np.array([[k,h],[h,k]])
        drift=np.block([[np.zeros((2,2)),np.eye(2)],[-K/m,(-gamma*np.eye(2)+B*np.array([[0.,1.],[-1.,0.]]))/m]])
        max_drift_real=max(max_drift_real,float(np.max(eigvals(drift).real)))
        min_covariance=min(min_covariance,float(np.min(np.linalg.eigvalsh(C))))
        independent=max(independent,abs(got-ref));closed_error=max(closed_error,abs(got-algebra),abs(base-base_algebra))
        energy_error=max(energy_error,abs(got+oracle.response(dict(e,bath=1),gamma)),abs(got-base+B*C[2,3]))
        parity_error=max(parity_error,abs(got-oracle.response(dict(e,field=-B),gamma)),abs(base-shortcut.response(dict(e,field=-B),gamma)))
        equal_temp=max(equal_temp,abs(oracle.response(dict(e,temperatures=[1.,1.]),gamma)))
        min_entropy=min(min_entropy,got*(1/T[1]-1/T[0]))
        zero=dict(e,field=0.);cal_difference=max(cal_difference,abs(oracle.response(zero,gamma)-shortcut.response(zero,gamma)))
        spots.append(dict(input=e,friction=gamma,oracle=got,frequency_reference=float(ref),shortcut=base))
    assert independent<1e-9 and closed_error<1e-9 and energy_error<1e-9
    assert parity_error<1e-9 and equal_temp<1e-9 and cal_difference<1e-9
    assert min_covariance>0 and max_drift_real<0 and min_entropy>=-1e-12
    for key,inputs_hidden in groups.items():
        model=oracle.Model();model.friction=reference.TRUE_PARAMETER
        assert np.max(abs(model.predict(inputs_hidden)-truth[key]))<1e-9
    controls={'oracle':local_control(TASK/'solution/model.py','oracle'),
              'shortcut':local_control(ROOT/'scripts/magnetic_bath_transfer_baseline.py','shortcut')}
    assert controls['oracle']['exit_code']==0 and controls['shortcut']['exit_code']==1
    assert '3 failed, 5 passed' in controls['shortcut']['stdout']
    assert '8 passed' in controls['oracle']['stdout']
    files=[p for p in TASK.rglob('*') if p.is_file() and '__pycache__' not in str(p) and '.pytest_cache' not in str(p)]
    files += [ROOT/'scripts/magnetic_bath_transfer_baseline.py',Path(__file__).resolve()]
    report=dict(status='staged_unevaluated',nominal=nominal,record_count=len(records),sigma=sigma,
      seeds={'calibration':78113,'noise':78117,'off_grid':78121},noise_trials=noise,
      noise_summary={'count':256,'calibration_passes':256,'oracle_passes':256,'shortcut_rejections':256,
        'max_oracle_error':max(max(r['hidden'][0].values()) for r in noise),
        'min_shortcut_error':min(min(r['hidden'][1].values()) for r in noise),
        'max_chi2':max(max(r['chi2']) for r in noise)},extrema=extrema,
      checks={'frequency_reference_max_error':independent,'closed_moment_max_error':closed_error,
        'first_law_and_channel_balance_max_error':energy_error,'field_reversal_max_error':parity_error,
        'equal_temperature_max_error':equal_temp,'zero_field_calibration_max_difference':cal_difference,
        'minimum_covariance_eigenvalue':min_covariance,'largest_drift_real_part':max_drift_real,
        'minimum_entropy_production':min_entropy,'global_calibration_slope_margin':monotonicity_margin,
        'maximum_noiseless_parameter_error_41_values':max(calibration_recovery),
        'profile_grid':grid.tolist(),'profile_loss':loss,'off_grid_checks':spots},
      local_controls=controls,source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    output=ROOT/'results/magnetic-bath-transfer-validation.json';output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'nominal':nominal,'noise':report['noise_summary'],'checks':{k:v for k,v in report['checks'].items() if k not in ('profile_grid','profile_loss','off_grid_checks')},'output':str(output)},indent=2))

if __name__=='__main__':main()
