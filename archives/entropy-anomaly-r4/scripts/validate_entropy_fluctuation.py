"""Scientific and local controls for staged entropy-anomaly revision4."""
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
from scipy.linalg import eigvals

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/entropy-anomaly'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def rms(a,b):return float(np.sqrt(np.mean((a-b)**2)/np.mean(b**2)))


def science(good,bad,ref):
    hidden=ref.hidden_inputs();experiments=sum(hidden.values(),[])
    agreement=[];refinement=[];mass_refinement=[];positive=[];calorimetry=[]
    for e in experiments:
        model=good.Model();model.friction=ref.TRUE_PARAMETER
        a=model.predict([e])[0];b=ref.predict([e])[0]
        fine=ref.predict([e],points=63,modes=40)[0]
        half=ref.predict([e],epsilon=.0025)[0]
        agreement.append(abs(b/a-1));refinement.append(abs((b-fine)/a));mass_refinement.append(abs((b-half)/a));positive.append(a)
    assert max(agreement)<3e-6 and max(refinement)<1e-8 and max(mass_refinement)<3e-6 and min(positive)>0
    corners=[];density=[];kinetic_variance=[]
    for g,t,c,w in itertools.product([.7,1.6],[.8,1.4],[.02,.65],[1,3]):
        e=dict(temperature=t,contrast=c,wavenumber=w,statistic='variance')
        a=good._rates(e,g)[1];b=ref.predict([e],g,points=63,modes=40,epsilon=.0025)[0]
        corners.append(abs(b/a-1))
        mass=.00125*g*g/(t*w*w)
        row=ref.finite_mass(t,c,w,g,mass,63,40)
        density.append(row['density_min']);kinetic_variance.append(row['variance'])
        calorimetry.append(abs(row['mean']-row['calorimetric_mean']))
        assert bad._rates(e,g)[1]>0
    assert max(corners)<3e-6 and min(density)>0 and min(kinetic_variance)>0 and max(calorimetry)<1e-8
    identities=[];small=[]
    for g in [.7,1.,1.6]:
        for c in [.1,.4,.65]:
            e=dict(temperature=1.2,contrast=c,wavenumber=2,statistic='mean')
            exact=1.2*4*(1-np.sqrt(1-c*c))/(2*g)
            identities.append(abs(good._rates(e,g)[0]-exact))
            assert good._rates(e,g)[0]==bad._rates(e,g)[0]
            assert abs(good._rates(e,g)[1]-bad._rates(e,g)[1]-2*exact)<1e-12
        zero=dict(temperature=1.,contrast=0.,wavenumber=1,statistic='variance')
        assert good._rates(zero,g)==bad._rates(zero,g)==(0.,0.)
        assert ref.predict([zero],g)[0]==0
    for c in [.1,.05,.025]:
        e=dict(temperature=1.,contrast=c,wavenumber=1,statistic='variance')
        mu,var=good._rates(e,1);closure=bad._rates(e,1)[1]
        small.append(dict(contrast=c,variance_over_twice_mean=var/(2*mu),shortcut_over_contrast4=closure/c**4))
    assert small[-1]['variance_over_twice_mean']<1.00003 and abs(small[-1]['shortcut_over_contrast4']-1/64)<1e-5
    # This is the limiting anomalous functional's fluctuation symmetry; no claim
    # about bare finite-mass calorimetry's full exponential-moment domain.
    x,D=ref.derivative(63);T=1+.6*np.cos(x);a=.36*np.sin(x)**2/(2*T)
    L=(ref.diags(T)@D@D).toarray()
    scgf={str(q):float(max(eigvals(L+np.diag(q*(1+q)*a)).real)) for q in [-1.,0.,-.8,-.2,.2,-1.2]}
    symmetry=max(abs(scgf['-0.8']-scgf['-0.2']),abs(scgf['0.2']-scgf['-1.2']))
    assert symmetry<1e-10 and abs(scgf['-1.0'])<1e-10 and abs(scgf['0.0'])<1e-10
    return dict(hidden_oracle_reference_relative_max=max(agreement),reference_resolution_relative_max=max(refinement),mass_halving_relative_max=max(mass_refinement),public_corner_reference_relative_max=max(corners),minimum_stationary_density=min(density),minimum_finite_mass_variance=min(kinetic_variance),finite_mass_calorimetric_cubic_mean_absolute_max=max(calorimetry),analytic_mean_absolute_max=max(identities),small_amplitude=small,limiting_anomaly_scgf=scgf,limiting_anomaly_symmetry_error=symmetry,minimum_hidden_rate=min(positive))


def controls():
    output={}
    for label,path in [('oracle',TASK/'solution/model.py'),('shortcut',ROOT/'scripts/entropy_anomaly_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='entropy-fluctuation-local-') as directory:
            work=Path(directory);shutil.copytree(TASK/'environment',work/'app');shutil.copytree(TASK/'tests',work/'tests');shutil.copy2(path,work/'app/model.py')
            env=dict(os.environ,PYTHONPATH=str(work/'app'),MODEL_PATH=str(work/'app/model.py'),METRICS_PATH=str(work/'metrics.json'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            start=time.perf_counter();result=subprocess.run([sys.executable,'-m','pytest','-q',str(work/'app/test_public.py'),str(work/'tests/test_hidden.py')],cwd=work,env=env,capture_output=True,text=True)
            output[label]=dict(returncode=result.returncode,seconds=time.perf_counter()-start,stdout=result.stdout,stderr=result.stderr,metrics=json.loads((work/'metrics.json').read_text()))
            assert result.returncode==(0 if label=='oracle' else 1)
            assert ('9 passed' in result.stdout if label=='oracle' else '3 failed, 6 passed' in result.stdout)
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');args=parser.parse_args();start=time.perf_counter()
    good=load(TASK/'solution/model.py','oracle');bad=load(ROOT/'scripts/entropy_anomaly_baseline.py','shortcut');ref=load(TASK/'tests/reference.py','reference')
    meta=json.loads((TASK/'tests/metadata.json').read_text());sigma=meta['sigma'];inputs=ref.calibration_inputs()
    physical=good.Model();physical.friction=ref.TRUE_PARAMETER;truth=physical.predict(inputs)
    if args.generate:
        values=truth+np.random.default_rng(meta['calibration_seed']).normal(0,sigma,len(inputs))
        records=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,values)]
        for side in ['environment','tests']:(TASK/side/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text());assert records==json.loads((TASK/'tests/data/calibration.json').read_text())
    assert [r['input'] for r in records]==inputs and all(r['sigma']==sigma for r in records)
    hidden=ref.hidden_inputs();expected={key:ref.predict(es) for key,es in hidden.items()};diagnostic=[key for key in hidden if key!='mean_anchors']
    def score(model):return {key:rms(model.predict(es),expected[key]) for key,es in hidden.items()}
    def chi(model,rows):return float(np.sum(((model.predict(inputs)-np.array([r['value'] for r in rows]))/sigma)**2)/(len(rows)-1))
    report=dict(task='entropy-anomaly',revision=4,status='staged_unevaluated',metadata=meta,calibration_records=len(records),controls={})
    for label,module in [('oracle',good),('shortcut',bad)]:
        model=module.Model().fit(records);scores=score(model)
        report['controls'][label]=dict(friction=model.friction,calibration_chi2=chi(model,records),hidden=scores)
        assert chi(model,records)<1.5 and abs(model.friction/ref.TRUE_PARAMETER-1)<.03 and scores['mean_anchors']<.04
        assert max(scores.values())<.04 if label=='oracle' else min(scores[k] for k in diagnostic)>.04
    parameters=[];chis=[];oracle_errors=[];shortcut_errors=[];rng=np.random.default_rng(meta['noise_validation_seed'])
    for _ in range(256):
        rows=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,truth+rng.normal(0,sigma,len(inputs)))]
        a,b=good.Model().fit(rows),bad.Model().fit(rows);assert a.friction==b.friction
        parameters.append(a.friction);chis.append(chi(a,rows));oracle_errors.append(max(score(a).values()));shortcut_errors.append(min(score(b)[k] for k in diagnostic))
        assert chis[-1]<1.5 and abs(a.friction/ref.TRUE_PARAMETER-1)<.03 and oracle_errors[-1]<.04 and shortcut_errors[-1]>.04 and score(b)['mean_anchors']<.04
    unit=good.Model().predict(inputs);information=float(np.sum(unit**2/sigma**2));assert information>1e5
    recover=[]
    for gamma in np.linspace(.7,1.6,25):
        m=good.Model();m.friction=float(gamma)
        rows=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,m.predict(inputs))]
        recover.append(abs(good.Model().fit(rows).friction/gamma-1))
    assert max(recover)<1e-12
    report['noise']=dict(realizations=256,calibration_passes=256,parameter_passes=256,oracle_passes=256,shortcut_rejections=256,maximum_chi2=max(chis),maximum_relative_parameter_error=float(max(abs(np.array(parameters)/ref.TRUE_PARAMETER-1))),parameter_range=[min(parameters),max(parameters)],oracle_hidden_max=max(oracle_errors),shortcut_hidden_min=min(shortcut_errors))
    report['identifiability']=dict(inverse_friction_information=information,noiseless_recovery_relative_max=max(recover),reason='Every nonuniform mean measurement is a known strictly positive coefficient divided by friction, so the full calibration uniquely determines friction on the positive range.')
    report['science']=science(good,bad,ref);report['local_controls']=controls();report['seconds']=time.perf_counter()-start
    files=[p for p in TASK.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]+[Path(__file__),ROOT/'scripts/entropy_anomaly_baseline.py']
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    (ROOT/'results/entropy-anomaly-r4-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
