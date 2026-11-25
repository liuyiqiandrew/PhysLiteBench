"""Author science and isolated local controls; no model or Docker evaluations."""
import argparse,ast,hashlib,importlib.util,json,os,shutil,subprocess,sys,tempfile,time,traceback
from itertools import product
from pathlib import Path
import numpy as np
from scipy.linalg import solve_continuous_lyapunov

BASE=Path(__file__).resolve().parents[1];TASK=BASE/'tasks/rotating-reservoir';RESULTS=BASE/'results';REPORT={}


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


oracle=load('rotating_oracle',TASK/'solution/model.py')
shortcut=load('rotating_source',BASE/'scripts/rotating_reservoir_baseline.py')
reference=load('rotating_reference',TASK/'tests/reference.py')
metadata=json.loads((TASK/'tests/metadata.json').read_text())
B=.9


def error(a,b):return float(np.sqrt(np.mean((a-b)**2)/np.mean(b*b)))


def physical_checks(e,g):
    C=oracle.stationary_covariance(**e,drag=g);tau=e['contact_time'];h=B/tau;s=e['strain_rate'];w=e['angular_speed']
    A=np.array([[s,-w],[w,-s]])
    physical=oracle.heat_rate(e,g);source=shortcut.heat_rate(e,g)
    stress=tau/B*np.trace(C[4:,4:]@A.T)
    qa=g*(np.trace(C[2:4,2:4])-2*e['temperature_a'])
    drive=np.trace(C[4:,:2]@A.T)+stress
    drift,noise=reference.matrices(e,g)
    transform=np.eye(6);transform[4:,:2]=np.eye(2);transform[4:,4:]=np.eye(2)/h
    bead=transform@C@transform.T
    return {'maximum_drift_real_part':float(np.linalg.eigvals(drift).real.max()),'minimum_covariance_eigenvalue':float(np.linalg.eigvalsh(C).min()),
        'heat':float(physical),'source':float(source),'flow_power':float(drive),'heat_a':float(qa),'elastic_flow_power':float(stress),
        'heat_identity_error':float(abs(physical-source-stress)),'first_law_error':float(abs(physical+qa-drive)),
        'entropy_rate':float(qa/e['temperature_a']+physical/e['temperature_b']),
        'independent_coordinate_drift_residual':float(abs(drift@bead+bead@drift.T+noise@noise.T).max())}


def local_controls():
    result={}
    for label,path in [('oracle',TASK/'solution/model.py'),('shortcut',BASE/'scripts/rotating_reservoir_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='rotating-r2-local-') as tmp:
            p=Path(tmp);shutil.copytree(TASK/'environment',p/'app');shutil.copytree(TASK/'tests',p/'tests');shutil.copy2(path,p/'app/model.py')
            env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(p/'app'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            start=time.monotonic();r=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',str(p/'app/test_public.py'),str(p/'tests/test_hidden.py')],cwd=p,env=env,capture_output=True,text=True,timeout=60)
            result[label]={'returncode':r.returncode,'seconds':time.monotonic()-start,'stdout':r.stdout,'stderr':r.stderr}
    (RESULTS/'rotating-reservoir-r2-local-controls.json').write_text(json.dumps(result,indent=2)+'\n')
    assert result['oracle']['returncode']==0 and '7 passed' in result['oracle']['stdout']
    assert result['shortcut']['returncode']==1 and '3 failed, 4 passed' in result['shortcut']['stdout']
    return {k:{x:y for x,y in v.items() if x not in ['stdout','stderr']} for k,v in result.items()}


def run(generate=False):
    start=time.monotonic();true=reference.TRUE_PARAMETER;sigma=metadata['measurement_sigma'];settings=reference.calibration_inputs();inputs=settings*metadata['calibration_repeats']
    clean=reference.predict(inputs,true)
    if generate:
        values=clean+np.random.default_rng(metadata['calibration_seed']).normal(0,sigma,len(inputs))
        records=[{'input':e,'value':float(y),'sigma':sigma} for e,y in zip(inputs,values)];text=json.dumps(records,indent=2)+'\n'
        for d in ['environment','tests']:(TASK/d/'data/calibration.json').write_text(text)
    records=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert [r['input'] for r in records]==inputs and len(settings)==20 and len(records)==320
    assert all(r['sigma']==sigma for r in records)
    assert (TASK/'environment/data/calibration.json').read_bytes()==(TASK/'tests/data/calibration.json').read_bytes()
    REPORT.update(status='science_in_progress',revision=2,model_runs=0,docker_runs=0,true_drag=true,calibration_count=len(records),distinct_settings=len(settings),sigma=sigma,
        calibration_seed=metadata['calibration_seed'],noise_seed=metadata['noise_seed'],prediction_limit=.04)
    groups=reference.hidden_inputs();truth={k:reference.predict(v,true) for k,v in groups.items()}
    actual={}
    for name,mod in [('oracle',oracle),('shortcut',shortcut)]:
        m=mod.Model().fit(records);residual=(m.predict(inputs)-np.array([r['value'] for r in records]))/sigma
        actual[name]={'drag':m.drag,'relative_parameter_error':abs(m.drag/true-1),'calibration_chi2':float(residual@residual/(len(records)-1)),
            'hidden':{k:error(m.predict(v),truth[k]) for k,v in groups.items()}}
        assert actual[name]['relative_parameter_error']<.03 and actual[name]['calibration_chi2']<1.5
    REPORT['actual_data']=actual
    referr=max(float(abs(oracle.predict_at(v,true)-truth[k]).max()) for k,v in groups.items())
    refined=max(float(abs(reference.predict(v,true,2e-12)-truth[k]).max()) for k,v in groups.items())
    bias=float(abs(oracle.predict_at(inputs,true)-clean).max()/sigma)
    REPORT['scored_reference']={'maximum_absolute_error':referr,'maximum_tolerance_refinement_change':refined,'maximum_calibration_bias_in_sigma':bias,
        'minimum_diagnostic_physical_heat':float(min(truth[k].min() for k in groups if k!='rotation_anchor')),'all_scored_cases_reference_checked':True}
    assert referr<1e-9 and refined<1e-9 and bias<.001
    def functions(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    assert functions(TASK/'environment/model.py')==functions(BASE/'scripts/rotating_reservoir_baseline.py')
    max_chi=max_parameter=max_oracle=max_anchor=0.;min_source=float('inf');fits=[];noise_rows=[]
    rng=np.random.default_rng(metadata['noise_seed'])
    for trial in range(metadata['noise_trials']):
        y=clean+rng.normal(0,sigma,len(clean));rows=[{'input':e,'value':float(v),'sigma':sigma} for e,v in zip(inputs,y)];entry={'index':trial}
        for name,mod in [('oracle',oracle),('shortcut',shortcut)]:
            m=mod.Model().fit(rows);fits.append(m.drag);chi=float(np.sum(((m.predict(inputs)-y)/sigma)**2)/(len(inputs)-1));pe=abs(m.drag/true-1)
            hidden={k:error(m.predict(v),truth[k]) for k,v in groups.items()};entry[name]={'drag':m.drag,'chi2':chi,'parameter_error':pe,'hidden':hidden}
            max_chi=max(max_chi,chi);max_parameter=max(max_parameter,pe);assert chi<1.5 and pe<.03
            for k,e in hidden.items():
                if name=='oracle':max_oracle=max(max_oracle,e);assert e<.04
                elif k=='rotation_anchor':max_anchor=max(max_anchor,e);assert e<.04
                else:min_source=min(min_source,e);assert e>.04
        noise_rows.append(entry)
    REPORT['noise']={'trials':metadata['noise_trials'],'all_expected_outcomes':True,'fit_min':min(fits),'fit_max':max(fits),'maximum_chi2':max_chi,'maximum_parameter_error':max_parameter,
        'maximum_oracle_error':max_oracle,'minimum_shortcut_diagnostic_error':min_source,'maximum_shortcut_anchor_error':max_anchor,'rows':noise_rows}
    recoveries=[];shared=calref=0.;min_gaps={k:float('inf') for k in groups if k!='rotation_anchor'};min_signals=[float('inf'),float('inf')]
    for g in np.linspace(.4,1.1,41):
        y=oracle.predict_at(settings,float(g));rows=[{'input':e,'value':float(v),'sigma':sigma} for e,v in zip(settings,y)]
        estimates=[mod.Model().fit(rows).drag for mod in [oracle,shortcut]]
        recoveries.append({'truth':float(g),'estimates':estimates,'maximum_error':max(abs(x-g) for x in estimates)})
        shared=max(shared,float(abs(y-shortcut.predict_at(settings,g)).max()))
        if g in [.4,1.1]:calref=max(calref,float(abs(y-reference.predict(settings,float(g))).max()))
        for k,v in groups.items():
            if k=='rotation_anchor':continue
            physical=oracle.predict_at(v,float(g));source=shortcut.predict_at(v,float(g));min_gaps[k]=min(min_gaps[k],error(source,physical))
            min_signals[0]=min(min_signals[0],float(physical.min()));min_signals[1]=min(min_signals[1],float(source.min()))
    # Exact monotonic static-flow anchor, with an analytic conservative derivative bound.
    min_derivative=.6*B*2*(B/1.1**2-.2)/(2.8*.2**2+(1.1+B)*.2+1+B/.4)**2
    REPORT['identifiability']={'proof':'For an Omega=strain=0 calibration setting, heat=(Ta−Tb)*sum_i b/[k_i*tau^2+(gamma+b)*tau+1+b/gamma]. At tau=.2 and positive known Ta−Tb=.6, every derivative is strictly positive because b/gamma^2−tau>=b/1.1^2−.2>0. Hence this included setting alone globally identifies gamma. Other settings enrich the fit.',
        'conservative_anchor_derivative_lower_bound':min_derivative,'recoveries':recoveries,'maximum_noiseless_fit_error':max(r['maximum_error'] for r in recoveries),
        'maximum_calibration_closure_difference':shared,'maximum_endpoint_calibration_reference_error':calref,'minimum_group_gaps':min_gaps,
        'minimum_physical_and_shortcut_diagnostic_signals':min_signals}
    assert min_derivative>0 and max(r['maximum_error'] for r in recoveries)<1e-7
    assert shared<1e-10 and calref<1e-9 and min(min_gaps.values())>.15 and min(min_signals)>.04
    # These are additional physical checks; continuous stability is proved separately by exact rational certificate.
    corners=[]
    for g,w,s,t,kx,ky,ta,tb in product([.4,1.1],[-.6,.6],[-.4,.4],[.2,.8],[1.,1.6],[2.2,2.8],[.8,1.4],[.8,1.4]):
        e=reference.experiment(w,s,t,kx,ky,ta,tb);c=physical_checks(e,g);corners.append({'drag':g,'input':e,**c})
    random=np.random.default_rng(173059);interior=[]
    for _ in range(128):
        e=reference.experiment(random.uniform(-.6,.6),random.uniform(-.4,.4),random.uniform(.2,.8),random.uniform(1,1.6),random.uniform(2.2,2.8),random.uniform(.8,1.4),random.uniform(.8,1.4));g=float(random.uniform(.4,1.1))
        interior.append({'drag':g,'input':e,**physical_checks(e,g)})
    allchecks=corners+interior
    REPORT['domain_checks']={'corners':corners,'interior':interior,'maximum_drift_real_part':max(r['maximum_drift_real_part'] for r in allchecks),'minimum_covariance_eigenvalue':min(r['minimum_covariance_eigenvalue'] for r in allchecks),
        'minimum_entropy_rate':min(r['entropy_rate'] for r in allchecks),'maximum_first_law_error':max(r['first_law_error'] for r in allchecks),'maximum_heat_identity_error':max(r['heat_identity_error'] for r in allchecks),
        'maximum_independent_coordinate_drift_residual':max(r['independent_coordinate_drift_residual'] for r in allchecks)}
    assert REPORT['domain_checks']['maximum_drift_real_part']<0 and REPORT['domain_checks']['minimum_covariance_eigenvalue']>0
    assert REPORT['domain_checks']['minimum_entropy_rate']>-1e-10
    assert max(REPORT['domain_checks'][k] for k in ['maximum_first_law_error','maximum_heat_identity_error','maximum_independent_coordinate_drift_residual'])<1e-10
    reference_rows=[]
    selected=corners[::4]+interior[:32]
    for r in selected:
        e=r['input'];g=r['drag'];value=reference.predict([e],g)[0];refined=reference.predict([e],g,2e-12)[0]
        reference_rows.append({'drag':g,'input':e,'oracle':r['heat'],'reference':float(value),'absolute_error':float(abs(value-r['heat'])),'refinement_change':float(abs(value-refined))})
    REPORT['domain_reference']={'cases':len(reference_rows),'maximum_absolute_error':max(r['absolute_error'] for r in reference_rows),'maximum_refinement_change':max(r['refinement_change'] for r in reference_rows),'rows':reference_rows}
    assert REPORT['domain_reference']['maximum_absolute_error']<1e-8 and REPORT['domain_reference']['maximum_refinement_change']<1e-8
    limits=[];max_gibbs=0.;max_reversal=0.;max_rest_formula=0.
    for g,t,T,kx,ky in product([.4,1.1],[.2,.8],[.8,1.4], [1.,1.6],[2.2,2.8]):
        e=reference.experiment(0.,0.,t,kx,ky,T,T);C=oracle.stationary_covariance(**e,drag=g);expected=np.diag([T/kx,T/ky,T,T,B*T/t,B*T/t]);max_gibbs=max(max_gibbs,float(abs(C-expected).max()))
    for e in settings:
        if e['angular_speed']==0:
            g=.73;t=e['contact_time'];exact=sum(B*(e['temperature_a']-e['temperature_b'])/(k*t*t+(g+B)*t+1+B/g) for k in [e['stiffness_x'],e['stiffness_y']])
            max_rest_formula=max(max_rest_formula,abs(oracle.heat_rate(e,g)-exact))
    for k,v in groups.items():
        for e in v:
            reverse={**e,'angular_speed':-e['angular_speed']};max_reversal=max(max_reversal,abs(oracle.heat_rate(e,.73)-oracle.heat_rate(reverse,.73)))
    for w,s in [(.3,.4),(-.5,-.3)]:
        g=.73;A=np.array([[s,-w],[w,-s]]);drift=np.block([[np.zeros((2,2)),np.eye(2)],[-np.diag([1.2,2.5])+B*A,-(g+B)*np.eye(2)]])
        Q=np.diag([0.,0.,2*(g*1.2+B*.9),2*(g*1.2+B*.9)]);C=solve_continuous_lyapunov(drift,-Q);L=np.column_stack([-A,np.eye(2)]);instant=float(B*np.trace(L@C@L.T)-2*B*.9)
        rows=[]
        for t in [.1,.01,.001,.0001]:
            e=reference.experiment(w,s,t,1.2,2.5,1.2,.9);physical=oracle.heat_rate(e,g);source=shortcut.heat_rate(e,g)
            rows.append({'contact_time':t,'physical':float(physical),'shortcut':float(source),'instantaneous_limit_error':abs(physical-instant),'closure_gap':abs(physical-source)})
        assert all(a['instantaneous_limit_error']>b['instantaneous_limit_error'] for a,b in zip(rows,rows[1:]))
        limits.append({'angular_speed':w,'strain_rate':s,'instantaneous_heat':instant,'rows':rows,'scope':'Contact times below.2 are author limits, not scored controls.'})
    REPORT['limits']={'maximum_equilibrium_gibbs_error':max_gibbs,'maximum_rotation_reversal_error':max_reversal,'maximum_static_conductance_error':max_rest_formula,'vanishing_contact_time':limits}
    assert max(max_gibbs,max_reversal,max_rest_formula)<1e-10
    # Replay the algebraic certificate verifier: no eigenvalue sampling or optimizer is needed here.
    cert=subprocess.run([sys.executable,'-B',str(BASE/'development/exact_certificate.py')],capture_output=True,text=True,timeout=60)
    assert cert.returncode==0,cert.stdout+cert.stderr
    proof=json.loads((BASE/'development/exact-stability-certificate.json').read_text())
    REPORT['continuous_stability']={'status':proof['status'],'vertices':len(proof['vertices']),'uniform_margin':proof['uniform_margin'],'proof':proof['proof'],
        'certificate_sha256':hashlib.sha256((BASE/'development/exact-stability-certificate.json').read_bytes()).hexdigest()}
    REPORT['local_controls']=local_controls()
    REPORT['source_hashes']={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',BASE/'scripts/rotating_reservoir_baseline.py',Path(__file__)]}
    REPORT.update(status='science_and_local_controls_complete',seconds=time.monotonic()-start)
    (RESULTS/'rotating-reservoir-r2-validation.json').write_text(json.dumps(REPORT,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:REPORT[k] for k in ['status','actual_data','scored_reference','continuous_stability','local_controls','seconds']},indent=2))
    print(json.dumps({k:v for k,v in REPORT['noise'].items() if k!='rows'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--generate',action='store_true')
    try:run(p.parse_args().generate)
    except Exception:
        REPORT.update(status='author_check_failed',traceback=traceback.format_exc())
        (RESULTS/f'author-check-failure-{time.time_ns()}.json').write_text(json.dumps(REPORT,indent=2,allow_nan=False)+'\n')
        raise
