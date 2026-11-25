"""Check Drude plate-force references, calibration and both completed controls."""
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
from scipy.special import zeta

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/metal-plate-force'


def module(path):
    spec=importlib.util.spec_from_file_location(str(path),path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true')
    parser.add_argument('--noise-trials',type=int,default=256)
    args=parser.parse_args();started=time.time()
    good=module(TASK/'solution/model.py');bad=module(ROOT/'scripts/metal_plate_force_baseline.py');ref=module(TASK/'tests/reference.py')
    meta=json.loads((TASK/'tests/metadata.json').read_text());true=ref.TRUE_PARAMETER
    inputs=ref.calibration_inputs();sigmas=np.array([2e-6/e['separation']**4 for e in inputs]);exact=ref.predict(inputs,true)
    if args.generate:
        measured=exact+np.random.default_rng(meta['calibration_seed']).normal(size=len(inputs))*sigmas
        records=[dict(input=e,value=float(y),sigma=float(s)) for e,y,s in zip(inputs,measured,sigmas)]
        for directory in ['environment','tests']:(TASK/directory/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records==json.loads((TASK/'tests/data/calibration.json').read_text())
    assert [r['input'] for r in records]==inputs and np.array_equal([r['sigma'] for r in records],sigmas)
    hidden=ref.hidden_inputs();truth={name:ref.predict(es,true) for name,es in hidden.items()}
    def chi(p,rs):
        residual=(good.predict_at(inputs,p)-np.array([r['value'] for r in rs]))/sigmas
        return float(residual@residual/(len(rs)-1))
    def scores(source,p):
        return {name:float(np.sqrt(np.mean((source.predict_at(es,p)-truth[name])**2)/np.mean(truth[name]**2))) for name,es in hidden.items()}
    report={'revision':1,'calibration_seed':meta['calibration_seed'],'noise_seed':meta['noise_seed'],'noise_trials':args.noise_trials,'controls':{}}
    for label,source in [('oracle',good),('shortcut',bad)]:
        model=source.Model().fit(records);p=model.plasma_frequency
        result=dict(plasma_frequency=p,parameter_relative_error=abs(p/true-1),calibration_chi2=chi(p,records),hidden=scores(source,p))
        assert result['calibration_chi2']<1.5 and result['parameter_relative_error']<.03
        assert max(result['hidden'].values())<meta['prediction_limit'] if label=='oracle' else min(result['hidden'].values())>meta['prediction_limit']
        report['controls'][label]=result
    # Exact zero-T forward equivalence makes both fitted objectives identical.
    assert np.max(abs(good.predict_at(inputs,true)-bad.predict_at(inputs,true)))==0
    rng=np.random.default_rng(meta['noise_seed']);draws=[]
    for _ in range(args.noise_trials):
        sample=[dict(input=e,value=float(v),sigma=float(s)) for e,v,s in zip(inputs,exact+rng.normal(size=len(inputs))*sigmas,sigmas)]
        p=good.Model().fit(sample).plasma_frequency
        row=dict(plasma_frequency=p,calibration_chi2=chi(p,sample),parameter_relative_error=abs(p/true-1),oracle=scores(good,p),shortcut=scores(bad,p))
        assert row['calibration_chi2']<1.5 and row['parameter_relative_error']<.03
        assert max(row['oracle'].values())<meta['prediction_limit'] and min(row['shortcut'].values())>meta['prediction_limit']
        draws.append(row)
    report['noise']={'calibration_passes':len(draws),'parameter_passes':len(draws),'oracle_passes':len(draws),'shortcut_passes':0,
                     'maximum_oracle_error':max(max(d['oracle'].values()) for d in draws),
                     'minimum_shortcut_error':min(min(d['shortcut'].values()) for d in draws),
                     'maximum_chi2':max(d['calibration_chi2'] for d in draws),'plasma_range':[min(d['plasma_frequency'] for d in draws),max(d['plasma_frequency'] for d in draws)]}
    report['noise_realizations']=draws
    es=inputs[::4]+sum(hidden.values(),[])
    y=good.predict_at(es,true);independent=ref.predict(es,true)
    scaled=np.array([e['separation']**4 for e in es])
    checks={'pressure_vs_free_energy_derivative_max_scaled':float(max(abs(y-independent)*scaled)),
            'oracle72_to128_max_scaled':float(max(abs(y-good.predict_at(es,true,128))*scaled)),
            'reference128_to192_max_scaled':float(max(abs(independent-np.array([ref.pressure(e,true,192) for e in es]))*scaled)),
            'reference_step_refinement_max_scaled':float(max(abs(independent-np.array([ref.pressure(e,true,128,5e-5) for e in es]))*scaled))}
    assert max(checks.values())<1e-8
    corner_error=[]
    for p in [4.,9.]:
        cases=[ref.experiment(a,t,g) for a in [.5,2.] for t in [0.,.05,.7] for g in [.15,.8]]
        a=good.predict_at(cases,p);b=ref.predict(cases,p)
        assert np.all(a<0)
        corner_error.append(max(abs((a-b)/b)))
    checks['domain_corner_max_relative_error']=float(max(corner_error));assert max(corner_error)<1e-6
    static=[]
    for xi in [1e-4,1e-7,1e-10]:
        te,tm=ref.boundary_reflection(np.array([.1,1.,5.]),xi,true,.4)
        static.append(dict(frequency=xi,max_te_squared=float(max(te)),max_tm_squared_error=float(max(abs(tm-1)))))
    assert static[-1]['max_te_squared']<1e-10 and static[-1]['max_tm_squared_error']<1e-10
    checks['static_impedance_limit']=static
    e=ref.experiment(2.,.7,.4);classical=-.7*zeta(3)/(8*np.pi*2**3)
    checks['classical_pressure_relative_error']=abs(good.pressure(e,true)/classical-1);assert checks['classical_pressure_relative_error']<1e-4
    ideal=-np.pi**2/240
    checks['ideal_zero_temperature_limit_relative_error']=abs(good.pressure(ref.experiment(1.,0.,.4),1e6)/ideal-1)
    assert checks['ideal_zero_temperature_limit_relative_error']<1e-4
    limit=[good.pressure(ref.experiment(1.,t,.4),true) for t in [0.,.003,.001]]
    checks['positive_damping_zero_temperature_limit_pressures']=limit
    assert abs(limit[2]-limit[0])<abs(limit[1]-limit[0]) and abs(limit[2]/limit[0]-1)<.002
    # The entire calibration vector is sensitive to the one material parameter.
    grids=np.linspace(4.,9.,101)
    loss=[float(np.mean(((good.predict_at(inputs,p)-exact)/sigmas)**2)) for p in grids]
    checks['noiseless_loss_grid_minimum']=float(grids[np.argmin(loss)])
    checks['loss_grid_local_minima']=int(sum(loss[i]<loss[i-1] and loss[i]<loss[i+1] for i in range(1,len(loss)-1)))
    assert checks['loss_grid_local_minima']==1 and checks['noiseless_loss_grid_minimum']==true
    report['physical_checks']=checks
    local={}
    for label,source in [('oracle',TASK/'solution/model.py'),('shortcut',ROOT/'scripts/metal_plate_force_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='metal-plate-'+label+'-') as tmp:
            tmp=Path(tmp);shutil.copytree(TASK/'environment',tmp/'app');shutil.copytree(TASK/'tests',tmp/'tests');shutil.copy2(source,tmp/'app/model.py')
            start=time.time();run=subprocess.run([sys.executable,'-m','pytest','-q',str(tmp/'app/test_public.py'),str(tmp/'tests/test_hidden.py')],cwd=tmp,env={**os.environ,'PYTHONPATH':str(tmp/'app')},capture_output=True,text=True)
            local[label]={'returncode':run.returncode,'seconds':time.time()-start,'stdout':run.stdout,'stderr':run.stderr}
            assert run.returncode==(0 if label=='oracle' else 1)
            assert ('7 passed' if label=='oracle' else '3 failed, 4 passed') in run.stdout
    report['local_controls']=local
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*TASK.rglob('*'),ROOT/'scripts/metal_plate_force_baseline.py',Path(__file__)] if p.is_file() and not {'__pycache__','.pytest_cache'}.intersection(p.parts)}
    report['seconds']=time.time()-started
    (ROOT/'results/metal-plate-force-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'results/metal-plate-force-local-controls.json').write_text(json.dumps(local,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k in ['controls','noise','physical_checks','seconds']},indent=2))


if __name__=='__main__':main()
