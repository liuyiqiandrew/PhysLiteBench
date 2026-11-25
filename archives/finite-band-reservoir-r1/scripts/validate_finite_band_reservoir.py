"""Validate a staged finite-band reservoir contact quench; no agent runs."""
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

ROOT=Path(__file__).resolve().parents[1];TASK=ROOT/'tasks/finite-band-reservoir'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def rms(a,b):return float(np.sqrt(np.mean((a-b)**2)/np.mean(b**2)))


def science(good,bad,ref):
    proto=load(ROOT/'prototype/check.py','prototype');hidden=ref.hidden_inputs();experiments=sum(hidden.values(),[])
    agreement=[];refinement=[];quadrature=[];populations=[];bookkeeping=[];weights=[]
    for e in experiments:
        g=ref.TRUE_PARAMETER;V=g*e['contact_multiplier'];a=good._occupation(e['orbital_energy'],e['temperature'],V)
        b=ref.predict([e])[0];fine=ref.predict([e],length=256)[0]
        agreement.append(abs(a-b));refinement.append(abs(b-fine));populations.append(a)
        q=proto.spectral(e['orbital_energy'],e['temperature'],V,n=768)
        quadrature.append(abs(a-q['correct']));weights.append(abs(q['spectral_weight']-1))
        r=ref.finite_chain(e['orbital_energy'],e['temperature'],V,256)
        bookkeeping.extend([abs(r['energy']-r['initial_energy']),abs(r['total_particles']-r['initial_particles'])])
        assert 0<=r['eigen_population_min']<=r['eigen_population_max']<=1
    assert max(agreement)<2e-6 and max(refinement)<2e-6 and max(quadrature)<1e-10 and max(weights)<1e-10 and max(bookkeeping)<1e-9 and min(populations)>0
    corners=[];gap=[];equilibrium=[]
    for g,e,t,c in itertools.product([.65,.95],[0.,.8],[.2,.8],[1,3,4]):
        V=g*c;a=good._occupation(e,t,V);q=proto.spectral(e,t,V)
        b=ref.predict([dict(orbital_energy=e,temperature=t,contact_multiplier=c)],g,length=256)[0]
        corners.append(abs(a-b));assert 0<=a<=1 and 0<=bad._occupation(e,t,V)<=1
        r=ref.finite_chain(e,t,V,256);equilibrium.append(abs(r['shortcut']-bad._occupation(e,t,V)))
        assert len(q['bound'])==(0 if c==1 else 2)
        if q['bound']:gap.extend([abs(p['energy'])-2 for p in q['bound']])
    assert max(corners)<2e-6 and max(equilibrium)<1e-10 and min(gap)>.01
    symmetry=[]
    for T,V in itertools.product([.2,.5,.8],[1.95,2.4,3.8]):
        Z=(V*V-2)/(2*(V*V-1))
        symmetry.extend([abs(good._occupation(0,T,V)-(.5+Z*Z)),abs(bad._occupation(0,T,V)-.5)])
    assert max(symmetry)<1e-10
    inputs=ref.calibration_inputs()[0:9];slopes=[];exact=[]
    for g in np.linspace(.65,.95,41):
        for e in inputs:
            a=good._occupation(e['orbital_energy'],e['temperature'],g)
            exact.append(abs(a-bad._occupation(e['orbital_energy'],e['temperature'],g)))
            slopes.append((good._occupation(e['orbital_energy'],e['temperature'],g+1e-5)-good._occupation(e['orbital_energy'],e['temperature'],g-1e-5))/2e-5)
    assert max(exact)==0 and min(slopes)>0
    return dict(hidden_oracle_reference_absolute_max=max(agreement),thermodynamic_refinement_absolute_max=max(refinement),quadrature_192_to768_absolute_max=max(quadrature),spectral_sum_rule_error=max(weights),finite_chain_particle_energy_balance_absolute_max=max(bookkeeping),public_corner_reference_absolute_max=max(corners),complete_gibbs_finite_chain_absolute_max=max(equilibrium),minimum_bound_energy_distance_from_band=min(gap),particle_hole_symmetric_identity_absolute_max=max(symmetry),calibration_equivalence_absolute=max(exact),minimum_full_range_calibration_derivative=min(slopes),hidden_occupation_range=[min(populations),max(populations)])


def controls():
    output={}
    for label,path in [('oracle',TASK/'solution/model.py'),('shortcut',ROOT/'scripts/finite_band_reservoir_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='finite-band-local-') as directory:
            work=Path(directory);shutil.copytree(TASK/'environment',work/'app');shutil.copytree(TASK/'tests',work/'tests');shutil.copy2(path,work/'app/model.py')
            env=dict(os.environ,PYTHONPATH=str(work/'app'),MODEL_PATH=str(work/'app/model.py'),METRICS_PATH=str(work/'metrics.json'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            start=time.perf_counter();result=subprocess.run([sys.executable,'-m','pytest','-q',str(work/'app/test_public.py'),str(work/'tests/test_hidden.py')],cwd=work,env=env,capture_output=True,text=True)
            output[label]=dict(returncode=result.returncode,seconds=time.perf_counter()-start,stdout=result.stdout,stderr=result.stderr,metrics=json.loads((work/'metrics.json').read_text()))
            assert result.returncode==(0 if label=='oracle' else 1)
            assert ('9 passed' in result.stdout if label=='oracle' else '3 failed, 6 passed' in result.stdout)
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');args=parser.parse_args();start=time.perf_counter()
    good=load(TASK/'solution/model.py','oracle');bad=load(ROOT/'scripts/finite_band_reservoir_baseline.py','shortcut');ref=load(TASK/'tests/reference.py','reference')
    meta=json.loads((TASK/'tests/metadata.json').read_text());sigma=meta['sigma'];inputs=ref.calibration_inputs()
    physical=good.Model();physical.coupling_scale=ref.TRUE_PARAMETER;truth=physical.predict(inputs)
    if args.generate:
        values=truth+np.random.default_rng(meta['calibration_seed']).normal(0,sigma,len(inputs))
        records=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,values)]
        for side in ['environment','tests']:(TASK/side/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text());assert records==json.loads((TASK/'tests/data/calibration.json').read_text())
    assert [r['input'] for r in records]==inputs and all(r['sigma']==sigma for r in records)
    hidden=ref.hidden_inputs();expected={key:ref.predict(es) for key,es in hidden.items()};diagnostic=[key for key in hidden if key!='weak_contact_anchors']
    def score(model):return {key:rms(model.predict(es),expected[key]) for key,es in hidden.items()}
    def chi(model,rows):return float(np.sum(((model.predict(inputs)-np.array([r['value'] for r in rows]))/sigma)**2)/(len(rows)-1))
    report=dict(task='finite-band-reservoir',revision=1,status='staged_unevaluated',metadata=meta,calibration_records=len(records),controls={})
    for label,module in [('oracle',good),('shortcut',bad)]:
        model=module.Model().fit(records);scores=score(model)
        report['controls'][label]=dict(coupling_scale=model.coupling_scale,calibration_chi2=chi(model,records),hidden=scores)
        assert chi(model,records)<1.5 and abs(model.coupling_scale/ref.TRUE_PARAMETER-1)<.03 and scores['weak_contact_anchors']<.04
        assert max(scores.values())<.04 if label=='oracle' else min(scores[k] for k in diagnostic)>.04
    parameters=[];chis=[];oracle_errors=[];shortcut_errors=[];rng=np.random.default_rng(meta['noise_validation_seed'])
    for _ in range(256):
        rows=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,truth+rng.normal(0,sigma,len(inputs)))]
        a,b=good.Model().fit(rows),bad.Model().fit(rows);assert a.coupling_scale==b.coupling_scale
        parameters.append(a.coupling_scale);chis.append(chi(a,rows));oracle_errors.append(max(score(a).values()));shortcut_errors.append(min(score(b)[k] for k in diagnostic))
        assert chis[-1]<1.5 and abs(a.coupling_scale/ref.TRUE_PARAMETER-1)<.03 and oracle_errors[-1]<.04 and shortcut_errors[-1]>.04 and score(b)['weak_contact_anchors']<.04
    recovery=[]
    for g in np.linspace(.6501,.9499,11):
        m=good.Model();m.coupling_scale=float(g)
        rows=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,m.predict(inputs))]
        recovery.append(abs(good.Model().fit(rows).coupling_scale/g-1))
    assert max(recovery)<1e-7
    report['noise']=dict(realizations=256,calibration_passes=256,parameter_passes=256,oracle_passes=256,shortcut_rejections=256,maximum_chi2=max(chis),maximum_relative_parameter_error=float(max(abs(np.array(parameters)/ref.TRUE_PARAMETER-1))),parameter_range=[min(parameters),max(parameters)],oracle_hidden_max=max(oracle_errors),shortcut_hidden_min=min(shortcut_errors),noiseless_full_range_recovery_relative_max=max(recovery))
    report['science']=science(good,bad,ref);report['local_controls']=controls();report['seconds']=time.perf_counter()-start
    files=[p for p in TASK.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]+[Path(__file__),ROOT/'scripts/finite_band_reservoir_baseline.py']
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    (ROOT/'results/finite-band-reservoir-r1-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
