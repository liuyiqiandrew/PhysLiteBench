"""Validate clamped-sphere torque noise without any agent evaluations."""
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

ROOT=Path(__file__).resolve().parents[1];TASK=ROOT/'tasks/hot-brownian-rotation'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def rms(actual,truth):return float(np.sqrt(np.mean((actual-truth)**2)/np.mean(truth**2)))


def science(good,bad,ref):
    hidden=ref.hidden_inputs();experiments=sum(hidden.values(),[])
    agreement=[];refinement=[];impedance_error=[];balance=[];positive=[]
    for e in experiments:
        oracle=good.predict_at([e],ref.TRUE_PARAMETER)[0]
        reference=ref.predict([e])[0];fine=ref.predict([e],cells=768)[0]
        agreement.append(abs(reference/oracle-1));refinement.append(abs((reference-fine)/oracle));positive.append(oracle)
        k=json.dumps(e,sort_keys=True)
        first=ref.shell_response(k,ref.TRUE_PARAMETER,384);second=ref.shell_response(k,ref.TRUE_PARAMETER,768)
        exact=good.impedance(e,ref.TRUE_PARAMETER)
        impedance_error.append(abs(((4*second[1]-first[1])/3-exact)/exact))
        balance.append(abs(second[2]/second[1].real-1))
    assert max(agreement)<2e-6 and max(refinement)<2e-6 and max(impedance_error)<2e-6 and max(balance)<2e-9
    assert min(positive)>0
    corners=[];domain=[];fdt=[]
    for viscosity in [.7,1.4]:
        for radius in [.8,1.2]:
            for omega in [0.,.0001,.05,120.]:
                e=ref.experiment(radius=radius,ambient=1.2,rise=2.4,omega=omega)
                a=good.predict_at([e],viscosity)[0];b=ref.predict([e],viscosity,cells=768)[0]
                corners.append(abs(b/a-1));assert a>0 and bad.predict_at([e],viscosity)[0]>0
                k=json.dumps(e,sort_keys=True)
                values=[]
                for extent in [16.,32.]:
                    lo=ref.shell_response(k,viscosity,768,extent)[0];hi=ref.shell_response(k,viscosity,1536,extent)[0]
                    values.append((4*hi-lo)/3)
                domain.append(abs((values[1]-values[0])/a))
                uniform=dict(e,rise=0.)
                fdt.append(abs(good.predict_at([uniform],viscosity)[0]/(2*uniform['ambient']*good.impedance(e,viscosity).real)-1))
    assert max(corners)<3e-6 and max(domain)<3e-6 and max(fdt)<1e-14
    weights=[good.thermal_weight(w) for w in [0.,.01,.1,1.,10.,100.,1000.,10000.]]
    assert weights[0]==.75 and all(.75<=w<1 for w in weights) and np.all(np.diff(weights)>0) and weights[-1]>.99
    cal=ref.calibration_inputs();cal_equivalence=max(abs(good.predict_at(cal,1)-bad.predict_at(cal,1)))
    static_ref=max(abs(ref.predict(cal,1)/good.predict_at(cal,1)-1));assert cal_equivalence==0 and static_ref<1e-10
    # Independent local noise powers add linearly across the prescribed baths.
    e=ref.experiment();unit_background=ref.predict([dict(e,ambient=1.,rise=0.)])[0]
    unit_rise=ref.predict([dict(e,ambient=0.,rise=1.)])[0]
    superposition=abs(ref.predict([e])[0]-(e['ambient']*unit_background+e['rise']*unit_rise))
    assert superposition<1e-9
    return dict(oracle_reference_relative_max=max(agreement),reference_refinement_relative_max=max(refinement),impedance_reference_relative_max=max(impedance_error),discrete_dissipation_identity_relative_max=max(balance),public_corner_reference_relative_max=max(corners),outer_domain_refinement_relative_max=max(domain),uniform_temperature_fdt_relative_max=max(fdt),thermal_weights=weights,static_calibration_equivalence_absolute=cal_equivalence,static_reference_relative_error=static_ref,independent_bath_superposition_absolute_error=superposition,minimum_hidden_spectrum=min(positive))


def controls():
    out={}
    for label,path in [('oracle',TASK/'solution/model.py'),('shortcut',ROOT/'scripts/hot_brownian_rotation_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='hot-rotation-local-') as directory:
            work=Path(directory);shutil.copytree(TASK/'environment',work/'app');shutil.copytree(TASK/'tests',work/'tests');shutil.copy2(path,work/'app/model.py')
            env=dict(os.environ,PYTHONPATH=str(work/'app'),MODEL_PATH=str(work/'app/model.py'),METRICS_PATH=str(work/'metrics.json'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            start=time.perf_counter();result=subprocess.run([sys.executable,'-m','pytest','-q',str(work/'app/test_public.py'),str(work/'tests/test_hidden.py')],cwd=work,env=env,capture_output=True,text=True)
            out[label]=dict(returncode=result.returncode,seconds=time.perf_counter()-start,stdout=result.stdout,stderr=result.stderr,metrics=json.loads((work/'metrics.json').read_text()))
            assert result.returncode==(0 if label=='oracle' else 1)
            assert ('9 passed' in result.stdout if label=='oracle' else '3 failed, 6 passed' in result.stdout)
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');args=parser.parse_args();started=time.perf_counter()
    good=load(TASK/'solution/model.py','oracle');bad=load(ROOT/'scripts/hot_brownian_rotation_baseline.py','shortcut');ref=load(TASK/'tests/reference.py','reference')
    meta=json.loads((TASK/'tests/metadata.json').read_text());sigma=meta['sigma'];inputs=ref.calibration_inputs();truth=ref.predict(inputs)
    if args.generate:
        observed=truth+np.random.default_rng(meta['calibration_seed']).normal(0,sigma,len(inputs));records=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,observed)]
        for name in ['environment','tests']:(TASK/name/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text());assert records==json.loads((TASK/'tests/data/calibration.json').read_text())
    assert [r['input'] for r in records]==inputs and all(r['sigma']==sigma for r in records)
    hidden=ref.hidden_inputs();htruth={k:ref.predict(es) for k,es in hidden.items()};diagnostics=[k for k in hidden if k!='anchors']
    def score(m):return {k:rms(m.predict(es),htruth[k]) for k,es in hidden.items()}
    def chi(m,rows):return float(np.sum(((m.predict(inputs)-np.array([r['value'] for r in rows]))/sigma)**2)/(len(rows)-1))
    report=dict(task='hot-brownian-rotation',revision=1,status='staged_unevaluated',metadata=meta,calibration_records=len(records),controls={})
    for label,module in [('oracle',good),('shortcut',bad)]:
        m=module.Model().fit(records);s=score(m);report['controls'][label]=dict(viscosity=m.viscosity,calibration_chi2=chi(m,records),hidden=s)
        assert chi(m,records)<1.5 and abs(m.viscosity/1.07-1)<.03 and s['anchors']<.025
        assert max(s.values())<.025 if label=='oracle' else min(s[k] for k in diagnostics)>.025
    rng=np.random.default_rng(meta['noise_validation_seed']);parameters=[];chis=[];oracle_errors=[];shortcut_errors=[]
    for _ in range(256):
        rows=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,truth+rng.normal(0,sigma,len(inputs)))];a,b=good.Model().fit(rows),bad.Model().fit(rows)
        assert abs(a.viscosity-b.viscosity)<1e-12
        parameters.append(a.viscosity);chis.append(chi(a,rows));oracle_errors.append(max(score(a).values()));shortcut_errors.append(min(score(b)[k] for k in diagnostics))
        assert chis[-1]<1.5 and abs(a.viscosity/1.07-1)<.03 and oracle_errors[-1]<.025 and shortcut_errors[-1]>.025
        assert score(b)['anchors']<.025
    report['noise']=dict(realizations=256,calibration_passes=256,parameter_passes=256,oracle_passes=256,shortcut_rejections=256,maximum_chi2=max(chis),maximum_relative_parameter_error=float(max(abs(np.array(parameters)/1.07-1))),parameter_range=[min(parameters),max(parameters)],oracle_hidden_max=max(oracle_errors),shortcut_hidden_min=min(shortcut_errors))
    coefficient=good.predict_at(inputs,1);report['calibration_information']=float(np.sum(coefficient**2/sigma**2));assert report['calibration_information']>1e5
    report['science']=science(good,bad,ref);report['local_controls']=controls();report['seconds']=time.perf_counter()-started
    files=[p for p in TASK.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]+[Path(__file__),ROOT/'scripts/hot_brownian_rotation_baseline.py']
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    (ROOT/'results/hot-brownian-rotation-r1-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
