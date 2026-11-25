"""Author-only fixed-trap science and local controls; no agent evaluations."""
from pathlib import Path
import argparse
import importlib.util
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np

STAGE=Path(__file__).resolve().parents[1]
TASK=STAGE/'tasks/fixed-trap-memory'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def nrms(y,truth):return float(np.linalg.norm(y-truth)/np.linalg.norm(truth))


def local_test(model):
    with tempfile.TemporaryDirectory(prefix='fixed-trap-control-') as d:
        app=Path(d);shutil.copytree(TASK/'environment',app,dirs_exist_ok=True)
        shutil.copy2(model,app/'model.py');start=time.monotonic()
        run=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',str(app/'test_public.py'),str(TASK/'tests/test_hidden.py')],cwd=app,env={**os.environ,'PYTHONPATH':str(app),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},capture_output=True,text=True)
        return {'returncode':run.returncode,'elapsed_seconds':time.monotonic()-start,'stdout':run.stdout,'stderr':run.stderr}


def main():
    p=argparse.ArgumentParser();p.add_argument('--generate',action='store_true');args=p.parse_args();start=time.monotonic()
    oracle=load(TASK/'solution/model.py','oracle');base=load(STAGE/'scripts/fixed_trap_memory_baseline.py','baseline');ref=load(TASK/'tests/reference.py','reference')
    inputs=ref.calibration_inputs();clean=ref.predict(inputs,ref.TRUE_PARAMETER);sigma=np.asarray([ref.instrument_sigma(e) for e in inputs])
    if args.generate:
        rng=np.random.default_rng(192011)
        rows=[{'input':e,'value':float(y),'sigma':float(sd)} for e,y,sd in zip(inputs,clean+rng.normal(size=len(clean))*sigma,sigma)]
        text=json.dumps(rows,indent=2)+'\n'
        for f in [TASK/'environment/data/calibration.json',TASK/'tests/data/calibration.json']:f.write_text(text)
    rows=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert (TASK/'environment/data/calibration.json').read_bytes()==(TASK/'tests/data/calibration.json').read_bytes()
    om=oracle.Model().fit(rows);bm=base.Model().fit(rows)
    exact=oracle.predict_at(inputs,ref.TRUE_PARAMETER);alternative=base.predict_at(inputs,ref.TRUE_PARAMETER)
    cal_difference=float(np.max(abs(exact-alternative)));cal_bias=float(np.max(abs(exact-clean)/sigma))
    assert cal_difference<1e-9 and cal_bias<1e-7
    groups=ref.hidden_inputs();targets={};hidden={}
    for name,es in groups.items():
        truth=ref.predict(es,ref.TRUE_PARAMETER);targets[name]=truth
        eo=nrms(om.predict(es),truth);eb=nrms(bm.predict(es),truth)
        hidden[name]={'truth':truth.tolist(),'minimum_signal':float(truth.min()),'oracle_nrms':eo,'shortcut_nrms':eb}
        assert np.min(truth)>1 and eo<.04
        if name!='exact_anchors':assert eb>.04
        else:assert eb<.04
    rng=np.random.default_rng(192013);fits=[];chis=[];worst=0.;best=1e10;fit_difference=0.
    for _ in range(256):
        y=clean+rng.normal(size=len(clean))*sigma
        records=[{'input':e,'value':float(value),'sigma':float(sd)} for e,value,sd in zip(inputs,y,sigma)]
        a=oracle.Model().fit(records);b=base.Model().fit(records)
        fits.append(a.escape_scale);fit_difference=max(fit_difference,abs(a.escape_scale-b.escape_scale))
        for model in [a,b]:
            chi=float(np.sum(((model.predict(inputs)-y)/sigma)**2)/(len(y)-1));chis.append(chi)
            assert chi<1.5 and abs(model.escape_scale/ref.TRUE_PARAMETER-1)<.03
        for name,es in groups.items():
            truth=targets[name];eo=nrms(a.predict(es),truth);eb=nrms(b.predict(es),truth)
            worst=max(worst,eo);assert eo<.04
            if name!='exact_anchors':best=min(best,eb);assert eb>.04
            else:assert eb<.04
    corners=[]
    for n,p,c,s,scale in itertools.product([2,9],[.45,.7,1.],[1.,25.],[0.,.15,.5,1.],[.6,1.4]):
        o=np.asarray(oracle.passage_moments(n,p,c,s,scale));r=np.asarray(ref.passage_moments(n,p,c,s,scale));b=np.asarray(base.passage_moments(n,p,c,s,scale))
        err=float(np.max(abs(o-r)/r));shared=abs(o[0]-b[0])/o[0]
        assert err<1e-10 and shared<1e-11 and np.min(o)>0 and np.min(b)>0
        if c==1 or p==1 or s in [0.,1.]:assert np.max(abs(o-b)/o)<1e-10
        scaled=np.asarray(oracle.passage_moments(n,p,c,s,2*scale))*[2,4]
        assert np.max(abs(scaled-o)/o)<1e-13
        corners.append({'controls':[n,p,c,s,scale],'reference_relative_error':err,'mean_relative_difference':float(shared),'moments':o.tolist(),'shortcut_moments':b.tolist()})
    rng=np.random.default_rng(192017);random_checks=[]
    for _ in range(32):
        n=int(rng.integers(2,10));p=float(rng.uniform(.45,1));c=float(rng.uniform(1,25));s=float(rng.uniform(0,1));scale=float(rng.uniform(.6,1.4))
        o=np.asarray(oracle.passage_moments(n,p,c,s,scale));r=np.asarray(ref.passage_moments(n,p,c,s,scale));b=np.asarray(base.passage_moments(n,p,c,s,scale))
        err=float(np.max(abs(o-r)/r));assert err<1e-10 and o[1]>=b[1]-1e-9
        random_checks.append({'controls':[n,p,c,s,scale],'reference_relative_error':err,'fractional_variance_excess':float((o[1]-b[1])/o[1])})
    recovery=[]
    for scale in np.linspace(.6,1.4,33):
        y=oracle.predict_at(inputs,float(scale));b=base.predict_at(inputs,float(scale))
        assert np.max(abs(y-b)/y)<1e-11
        records=[{'input':e,'value':float(v),'sigma':float(sd)} for e,v,sd in zip(inputs,y,sigma)]
        fitted=base.Model().fit(records).escape_scale;assert abs(fitted-scale)<1e-6
        gaps={name:nrms(base.predict_at(es,float(scale)),oracle.predict_at(es,float(scale))) for name,es in groups.items()}
        assert all(gaps[name]>.04 for name in groups if name!='exact_anchors')
        recovery.append({'true_scale':float(scale),'fitted_scale':fitted,'absolute_error':abs(fitted-scale),'hidden_gaps':gaps})
    # Each calibrated response is a positive coefficient times scale^-1 or scale^-2.
    powers=np.asarray([1 if e['statistic']=='mean' else 2 for e in inputs])
    coefficient=oracle.predict_at(inputs,1.)
    scaling_error=max(float(np.max(abs(oracle.predict_at(inputs,float(q))-coefficient/q**powers))) for q in [.6,.83,1.21,1.4])
    assert scaling_error<1e-9
    sensitivity=-powers*exact/ref.TRUE_PARAMETER
    residual=(om.predict(inputs)-np.asarray([r['value'] for r in rows]))/sigma
    report={'status':'scientific_checks_passed','model_evaluations':0,'calibration':{'records':len(rows),'unique_inputs':len({json.dumps(e,sort_keys=True) for e in inputs}),'oracle_scale':om.escape_scale,'shortcut_scale':bm.escape_scale,'chi2':float(residual@residual)/(len(rows)-1),'maximum_control_absolute_difference':cal_difference,'maximum_reference_bias_in_sigma':cal_bias,'minimum_absolute_sensitivity':float(np.min(abs(sensitivity))),'fisher_information_at_truth':float(np.sum((sensitivity/sigma)**2)),'exact_scale_power_error':scaling_error,'uncertainty':'Fixed instrument sigma .1 for time means and .5 for time variances, independent of rate scale/clean response.'},'hidden':hidden,'noise256':{'all_calibration_and_parameter_pass':True,'all_oracle_pass':True,'all_shortcut_diagnostic_fail':True,'maximum_chi2':max(chis),'fitted_scale_range':[min(fits),max(fits)],'maximum_control_fit_difference':fit_difference,'maximum_oracle_hidden_error':worst,'minimum_shortcut_diagnostic_error':best},'domain_corners':corners,'random_reference_checks':random_checks,'parameter_recovery':recovery,'elapsed_science_seconds':time.monotonic()-start}
    (STAGE/'results/fixed-trap-memory-r1-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    controls={'oracle':local_test(TASK/'solution/model.py'),'shortcut':local_test(STAGE/'scripts/fixed_trap_memory_baseline.py')}
    (STAGE/'results/fixed-trap-memory-r1-local-controls.json').write_text(json.dumps(controls,indent=2)+'\n')
    assert controls['oracle']['returncode']==0
    assert controls['shortcut']['returncode']==1 and '3 failed, 4 passed' in controls['shortcut']['stdout']
    print(json.dumps({'calibration':report['calibration'],'noise256':report['noise256'],'local_controls':{k:{q:v[q] for q in ['returncode','elapsed_seconds']} for k,v in controls.items()}},indent=2))

if __name__=='__main__':main()
