"""Author validation and isolated Python controls; no Docker or model evaluations."""
import argparse,hashlib,importlib.util,json,os,shutil,subprocess,sys,tempfile,time,traceback
from itertools import product
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss

BASE=Path(__file__).resolve().parents[1];TASK=BASE/'tasks/hydrodynamic-heating';RESULTS=BASE/'results';REPORT={}

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

oracle=load('hydro_oracle',TASK/'solution/model.py')
source=load('hydro_source',BASE/'scripts/hydrodynamic_heating_baseline.py')
ref=load('hydro_reference',TASK/'tests/reference.py')
meta=json.loads((TASK/'tests/metadata.json').read_text())
LOADED={str(p.relative_to(BASE)):digest(p) for p in [TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',BASE/'scripts/hydrodynamic_heating_baseline.py',Path(__file__)]}


def rms_error(a,b):return float(np.sqrt(np.mean((a-b)**2)/np.mean(b*b)))

def records(inputs,y,sigma):return [dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,y)]

def control_metrics(mod,rows,truth,groups):
    model=mod.Model().fit(rows);p=model.plasma_frequency
    residual=(model.predict([r['input'] for r in rows])-np.array([r['value'] for r in rows]))/np.array([r['sigma'] for r in rows])
    return dict(parameter=p,relative_parameter_error=abs(p/ref.TRUE_PARAMETER-1),calibration_chi2=float(residual@residual)/(len(rows)-1),hidden={k:rms_error(model.predict(e),truth[k]) for k,e in groups.items()})


def optical_balance(e,p):
    controls={k:e[k] for k in ('frequency','thickness','friction','relaxation')}
    field=oracle.modal_fields(controls,p);d=e['thickness'];w=e['frequency'];alpha=e['friction'];tau=e['relaxation']
    electric,_,current,_=field([0,d]);tractions=alpha*current/(1-1j*w*tau)
    means=abs(tractions)**2/(alpha*p*p)
    x,weight=leggauss(80);_,_,j,jz=field((x+1)*d/2)
    bulk=d/2*np.dot(weight,oracle.GAMMA*abs(j)**2+oracle.NU*abs(jz)**2)/(p*p)
    absorption=1-abs(electric[0]-1)**2-abs(electric[1])**2
    return dict(absorption=float(absorption),bulk=float(bulk),contact_means=means.tolist(),residual=float(abs(absorption-bulk-sum(means))))


def local_controls():
    result={}
    for label,path in [('oracle',TASK/'solution/model.py'),('shortcut',BASE/'scripts/hydrodynamic_heating_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='hydro-r3-local-') as temp:
            p=Path(temp);shutil.copytree(TASK/'environment',p/'app');shutil.copytree(TASK/'tests',p/'tests');shutil.copy2(path,p/'app/model.py')
            env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(p/'app'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            start=time.monotonic();r=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',str(p/'app/test_public.py'),str(p/'tests/test_hidden.py')],cwd=p,env=env,capture_output=True,text=True,timeout=60)
            result[label]=dict(returncode=r.returncode,seconds=time.monotonic()-start,stdout=r.stdout,stderr=r.stderr)
    (RESULTS/'hydrodynamic-heating-r3-local-controls.json').write_text(json.dumps(result,indent=2)+'\n')
    assert result['oracle']['returncode']==0 and '7 passed' in result['oracle']['stdout']
    assert result['shortcut']['returncode']==1 and '3 failed, 4 passed' in result['shortcut']['stdout']
    return {k:{x:y for x,y in v.items() if x not in ['stdout','stderr']} for k,v in result.items()}


def run(generate):
    start=time.monotonic();sigma=meta['measurement_sigma'];settings=ref.calibration_inputs();inputs=settings*meta['calibration_repeats']
    clean=ref.predict(inputs)
    if generate:
        y=clean+np.random.default_rng(meta['calibration_seed']).normal(0,sigma,len(inputs));text=json.dumps(records(inputs,y,sigma),indent=2)+'\n'
        for d in ['environment','tests']:(TASK/d/'data/calibration.json').write_text(text)
    rows=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert [r['input'] for r in rows]==inputs and len(rows)==288 and all(r['sigma']==sigma for r in rows)
    assert (TASK/'environment/data/calibration.json').read_bytes()==(TASK/'tests/data/calibration.json').read_bytes()
    groups=ref.hidden_inputs();t0=time.monotonic();truth={k:ref.predict(v) for k,v in groups.items()}
    REPORT.update(status='in_progress',model_runs=0,docker_runs=0,loaded_source_sha256=LOADED,calibration_records=len(rows),fixed_sigma=sigma,cold_reference_seconds=time.monotonic()-t0)
    REPORT['actual_data']={name:control_metrics(mod,rows,truth,groups) for name,mod in [('oracle',oracle),('shortcut',source)]}
    noise=[];random=np.random.default_rng(meta['noise_seed'])
    for i in range(256):
        sample=records(inputs,clean+random.normal(0,sigma,len(inputs)),sigma)
        results={name:control_metrics(mod,sample,truth,groups) for name,mod in [('oracle',oracle),('shortcut',source)]}
        for r in results.values():assert r['relative_parameter_error']<.03 and r['calibration_chi2']<1.5
        assert max(results['oracle']['hidden'].values())<.04
        assert all(results['shortcut']['hidden'][k]>.04 for k in groups if k!='mean_anchor')
        assert results['shortcut']['hidden']['mean_anchor']<.04
        noise.append(dict(seed_index=i,**results))
    REPORT['noise']={'trials':256,'oracle_passes':256,'shortcut_rejections':256,'maximum_oracle_error':max(max(r['oracle']['hidden'].values()) for r in noise),'minimum_shortcut_diagnostic_error':min(r['shortcut']['hidden'][k] for r in noise for k in groups if k!='mean_anchor'),'rows':noise}
    # Dense full-parameter profiles plus independent-step derivatives. These are numerical global-range checks, not interval proofs.
    grid=np.linspace(.85,1.15,1001);curves=np.array([oracle.predict_at(settings,float(p)) for p in grid]);derivative=np.diff(curves,axis=0)/(grid[1]-grid[0])
    fits=[];margins=[]
    for p in np.linspace(.85,1.15,41):
        values=oracle.predict_at(settings,float(p));sample=records(settings,values,sigma)
        estimates=[m.Model().fit(sample).plasma_frequency for m in [oracle,source]]
        profile=np.sum((curves-values)**2,axis=1)
        minima=int(np.sum((profile[1:-1]<profile[:-2])&(profile[1:-1]<profile[2:])))
        fits.append(dict(truth=float(p),estimates=estimates,max_error=max(abs(x-p) for x in estimates),sampled_interior_objective_minima=minima))
        margins.append(dict(plasma_frequency=float(p),groups={k:rms_error(source.predict_at(e,p),oracle.predict_at(e,p)) for k,e in groups.items()},shared_mean_error=float(np.max(abs(oracle.predict_at(settings,p)-source.predict_at(settings,p)))) )
    checks=[]
    for p in [.85,1.,1.15]:
        for step in [1e-4,1e-5,1e-6]:checks.append(dict(p=p,step=step,derivative=((oracle.predict_at(settings,p+step)-oracle.predict_at(settings,p-step))/(2*step)).tolist()))
    REPORT['identifiability']=dict(qualification='Full-range dense monotonic profiles, derivative step checks and noiseless global-objective/recovery evidence; not a formal analytic proof.',parameter_grid=grid.tolist(),calibration_curves=curves.tolist(),minimum_secant_derivative=float(derivative.min()),derivative_checks=checks,noiseless_fits=fits,full_parameter_margins=margins)
    assert derivative.min()>.05 and all(min(c['derivative'])>.05 for c in checks)
    assert max(r['max_error'] for r in fits)<1e-7 and max(r['sampled_interior_objective_minima'] for r in fits)<=1
    assert min(r['groups'][k] for r in margins for k in groups if k!='mean_anchor')>.4
    # Every graded input, including mean anchors, is independently refined.
    scored=[]
    for name,experiments in groups.items():
        for e in experiments:
            value=float(ref.predict([e])[0]);fine=float(ref.predict([e],tolerance=2e-11)[0]);pred=float(oracle.predict_at([e],ref.TRUE_PARAMETER)[0])
            scored.append(dict(group=name,input=e,reference=value,oracle=pred,absolute_error=abs(value-pred),refinement=abs(value-fine)))
    REPORT['scored_reference']=dict(rows=scored,maximum_error=max(r['absolute_error'] for r in scored),maximum_refinement=max(r['refinement'] for r in scored))
    assert REPORT['scored_reference']['maximum_error']<1e-8 and REPORT['scored_reference']['maximum_refinement']<1e-8
    domain=[]
    controls=[(ref.experiment(w,d,a,t),p) for p,w,d,a,t in product([.85,1.15],[.7,1.5],[.3,1.2],[.08,.4],[.2,1.2])]
    random=np.random.default_rng(193059)
    controls += [(ref.experiment(random.uniform(.7,1.5),random.uniform(.3,1.2),random.uniform(.08,.4),random.uniform(.2,1.2)),random.uniform(.85,1.15)) for _ in range(32)]
    for e,p in controls:
        es=[dict(e,readout=r) for r in ['mean','in_phase','quadrature']];y=oracle.predict_at(es,p);other=ref.predict(es,p);fine=ref.predict(es,p,2e-11)
        q=complex(y[1],y[2]);wrong=source.predict_at(es,p);w=complex(wrong[1],wrong[2]);balance=optical_balance(e,p)
        domain.append(dict(input=e,plasma_frequency=float(p),oracle=y.tolist(),source=wrong.tolist(),reference=other.tolist(),reference_error=float(max(abs(y-other))),reference_refinement=float(max(abs(other-fine))),optical_balance=balance,mean_equality=float(abs(y[0]-wrong[0])),harmonic_identity=float(abs(w-(1-1j*e['frequency']*e['relaxation'])*q)),positive_heat_identity=float(abs(abs(q)-y[0]))))
    REPORT['domain']=dict(corners=32,interior=32,rows=domain,maximum_reference_error=max(r['reference_error'] for r in domain),maximum_reference_refinement=max(r['reference_refinement'] for r in domain),maximum_energy_residual=max(r['optical_balance']['residual'] for r in domain),minimum_mean=min(r['oracle'][0] for r in domain))
    assert REPORT['domain']['maximum_reference_error']<1e-8 and REPORT['domain']['maximum_reference_refinement']<1e-8
    assert REPORT['domain']['maximum_energy_residual']<1e-10
    assert max(max(r[k] for k in ['mean_equality','harmonic_identity','positive_heat_identity']) for r in domain)<1e-12
    limits=[]
    for key,values in [('relaxation',[.3,.03,.003,0.]),('friction',[.01,.001,.0001,100.,1000.,10000.])]:
        for value in values:
            e=ref.experiment(1.2,.8,.2,.7);e[key]=value;es=[dict(e,readout=r) for r in ['mean','in_phase','quadrature']]
            limits.append(dict(control=key,value=value,oracle=oracle.predict_at(es,1.03).tolist(),source=source.predict_at(es,1.03).tolist(),scope='unscored limit outside declared task domain'))
    REPORT['limits']=limits
    REPORT['local_controls']=local_controls()
    assert LOADED=={n:digest(BASE/n) for n in LOADED},'Loaded source changed during validation'
    REPORT.update(status='science_and_local_controls_complete',seconds=time.monotonic()-start)
    (RESULTS/'hydrodynamic-heating-r3-validation.json').write_text(json.dumps(REPORT,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:REPORT[k] for k in ['status','actual_data','cold_reference_seconds','local_controls','seconds']},indent=2))
    print(json.dumps({k:v for k,v in REPORT['noise'].items() if k!='rows'},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true')
    try:run(parser.parse_args().generate)
    except Exception:
        REPORT.update(status='failed_author_validation',traceback=traceback.format_exc());(RESULTS/'hydrodynamic-heating-r3-failed-validation.json').write_text(json.dumps(REPORT,indent=2,allow_nan=False)+'\n');raise
