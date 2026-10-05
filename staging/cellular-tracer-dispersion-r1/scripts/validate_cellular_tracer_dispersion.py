"""Author science and isolated controls, without Docker or model evaluations."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,ast,hashlib,importlib.util,json,shutil,subprocess,sys,tempfile,time,traceback
from itertools import product
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve,expm_multiply,eigs

BASE=Path(__file__).resolve().parents[1];TASK=BASE/'tasks/cellular-tracer-dispersion';RESULTS=BASE/'results';REPORT={}
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
O=load('cellular_oracle',TASK/'solution/model.py')
S=load('cellular_shortcut',BASE/'scripts/cellular_tracer_dispersion_baseline.py')
R=load('cellular_reference',TASK/'tests/reference.py')
META=json.loads((TASK/'tests/metadata.json').read_text())

def tensor(module,e,D,**kwargs):return module.diffusion_tensor(e['a'],e['b'],e['c'],e['d'],D,**kwargs)
def error(a,b):return float(np.linalg.norm(a-b)/np.linalg.norm(b))

def local_controls():
    results={}
    for label,path in [('oracle',TASK/'solution/model.py'),('shortcut',BASE/'scripts/cellular_tracer_dispersion_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='cellular-r1-control-') as tmp:
            p=Path(tmp);shutil.copytree(TASK/'environment',p/'app');shutil.copytree(TASK/'tests',p/'tests');shutil.copy2(path,p/'app/model.py')
            env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(p/'app'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            start=time.monotonic();r=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',str(p/'app/test_public.py'),str(p/'tests/test_hidden.py')],cwd=p,env=env,capture_output=True,text=True,timeout=60)
            results[label]={'returncode':r.returncode,'seconds':time.monotonic()-start,'stdout':r.stdout,'stderr':r.stderr}
    (RESULTS/'cellular-tracer-dispersion-r1-local-controls.json').write_text(json.dumps(results,indent=2)+'\n')
    assert results['oracle']['returncode']==0 and '7 passed' in results['oracle']['stdout']
    assert results['shortcut']['returncode']==1 and '3 failed, 4 passed' in results['shortcut']['stdout']
    return {k:{x:y for x,y in v.items() if x not in ['stdout','stderr']} for k,v in results.items()}

def finite_time(e,D,n=32):
    L,L1,U,second,minimum=R.generator(e['a'],e['b'],e['c'],e['d'],D,n);size=L.shape[0]
    average=np.asarray(L1[0].sum(axis=0)).ravel()/size
    aug=sparse.bmat([[L,None,sparse.csc_matrix(U[:,0,None])],[sparse.csc_matrix(2*average[None,:]),sparse.csc_matrix((1,1)),sparse.csc_matrix([[second[0,0]]])],[None,None,sparse.csc_matrix((1,1))]],format='csc')
    y=np.zeros(size+2);y[-1]=1;T=8/D
    vals=expm_multiply(aug,y,start=0,stop=2*T,num=3,traceA=float(aug.diagonal().sum()))[:,-2]
    slope=(vals[2]-vals[1])/(2*T)
    discrete=R.tensor_on_grid(e['a'],e['b'],e['c'],e['d'],D,n)[0,0]
    return {'input':e,'D':D,'grid':n,'times':[0,T,2*T],'variances':vals.tolist(),'late_interval_coefficient':float(slope),'stationary_same_grid':float(discrete),'relative_time_error':float(abs(slope-discrete)/discrete),'minimum_rate':minimum}

def counting_check(e,D,n=32):
    L,first,_,_,_=R.generator(e['a'],e['b'],e['c'],e['d'],D,n)
    # The two displacement-marked off-diagonals distinguish wrapping jumps.
    h=2*np.pi/n;ids=np.arange(n*n).reshape(n,n);x,y=np.indices((n,n))*h
    ux=(e['a']*np.sin(y)+e['c']*np.sin(2*y)).ravel();uy=(e['b']*np.sin(x)+e['d']*np.sin(2*x)).ravel()
    rows=[];cols=[];rates=[];marks=[]
    for axis,v in [(0,ux),(1,uy)]:
        for sign in [-1,1]:
            rows.extend(ids.ravel());cols.extend(np.roll(ids,-sign,axis=axis).ravel());rates.extend(D/h**2+sign*v/(2*h));marks.extend([sign*h if axis==0 else 0.]*(n*n))
    rates=np.array(rates);marks=np.array(marks);answers=[]
    for s in [.02,.01]:
        eigen=[]
        for sign in [-1,1]:
            tilted=sparse.coo_matrix((rates*np.exp(sign*s*marks),(rows,cols)),shape=L.shape).tocsc()+sparse.diags(L.diagonal())
            val=eigs(tilted,k=1,sigma=0,return_eigenvectors=False)[0];assert abs(val.imag)<1e-9;eigen.append(float(val.real))
        answers.append({'s':s,'eigenvalues':eigen,'coefficient':sum(eigen)/(2*s*s)})
    extrap=(4*answers[1]['coefficient']-answers[0]['coefficient'])/3
    exact=R.tensor_on_grid(e['a'],e['b'],e['c'],e['d'],D,n)[0,0]
    return {'input':e,'D':D,'grid':n,'rows':answers,'extrapolated':extrap,'same_grid_moment_coefficient':float(exact),'relative_error':float(abs(extrap-exact)/exact)}

def run(generate=False):
    start=time.monotonic();true=R.TRUE_PARAMETER;sig=META['measurement_sigma'];settings=R.calibration_inputs();inputs=settings*META['calibration_repeats'];clean=R.predict(inputs,true)
    if generate:
        y=clean+np.random.default_rng(META['calibration_seed']).normal(0,sig,len(clean));records=[{'input':e,'value':float(v),'sigma':sig} for e,v in zip(inputs,y)];text=json.dumps(records,indent=2)+'\n'
        for d in ['environment','tests']:(TASK/d/'data/calibration.json').write_text(text)
    records=json.loads((TASK/'environment/data/calibration.json').read_text());assert [r['input'] for r in records]==inputs
    assert len(settings)==24 and len(records)==288 and all(r['sigma']==sig for r in records)
    assert (TASK/'environment/data/calibration.json').read_bytes()==(TASK/'tests/data/calibration.json').read_bytes()
    REPORT.update(status='science_in_progress',model_runs=0,docker_runs=0,true_diffusivity=true,calibration_count=len(records),distinct_settings=len(settings),sigma=sig,calibration_seed=META['calibration_seed'],noise_seed=META['noise_seed'],prediction_limit=.04)
    groups=R.hidden_inputs();truth={k:R.predict(v,true) for k,v in groups.items()}
    actual={}
    for name,mod in [('oracle',O),('shortcut',S)]:
        m=mod.Model().fit(records);res=(m.predict(inputs)-np.array([r['value'] for r in records]))/sig
        actual[name]={'diffusivity':m.diffusivity,'parameter_error':abs(m.diffusivity/true-1),'chi2':float(res@res/(len(inputs)-1)),'hidden':{k:error(m.predict(v),truth[k]) for k,v in groups.items()}}
        assert actual[name]['parameter_error']<.03 and actual[name]['chi2']<1.5
    REPORT['actual_data']=actual
    reference_rows=[]
    for group,es in [('calibration',settings),*groups.items()]:
        for e in es:
            physical=O.predict_at([e],true)[0];ref=R.predict([e],true)[0];refined=R.predict([e],true,96)[0]
            reference_rows.append({'group':group,'input':e,'oracle':float(physical),'reference48_96':float(ref),'reference96_192':float(refined),'absolute_error':float(abs(ref-physical)),'refinement_change':float(abs(ref-refined))})
    REPORT['scored_reference']={'all_calibration_and_hidden_inputs_checked':True,'rows':reference_rows,'maximum_absolute_error':max(r['absolute_error'] for r in reference_rows),'maximum_refinement_change':max(r['refinement_change'] for r in reference_rows),'maximum_calibration_bias_sigma':float(abs(O.predict_at(inputs,true)-clean).max()/sig)}
    assert REPORT['scored_reference']['maximum_absolute_error']<1e-4 and REPORT['scored_reference']['maximum_refinement_change']<1e-4
    assert REPORT['scored_reference']['maximum_calibration_bias_sigma']<.01
    def functions(p):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(p.read_text()).body if isinstance(x,ast.FunctionDef)}
    public=functions(TASK/'environment/model.py');completed=functions(BASE/'scripts/cellular_tracer_dispersion_baseline.py');assert public==completed
    rng=np.random.default_rng(META['noise_seed']);noise=[]
    for i in range(META['noise_trials']):
        y=clean+rng.normal(0,sig,len(clean));rows=[{'input':e,'value':float(v),'sigma':sig} for e,v in zip(inputs,y)];entry={'index':i}
        for name,mod in [('oracle',O),('shortcut',S)]:
            m=mod.Model().fit(rows);chi=float(np.sum(((m.predict(inputs)-y)/sig)**2)/(len(inputs)-1));pe=abs(m.diffusivity/true-1);hidden={k:error(m.predict(v),truth[k]) for k,v in groups.items()}
            entry[name]={'diffusivity':m.diffusivity,'chi2':chi,'parameter_error':pe,'hidden':hidden};assert chi<1.5 and pe<.03
            assert all(err<.04 if name=='oracle' or k=='anchors' else err>.04 for k,err in hidden.items())
        noise.append(entry)
    REPORT['noise']={'trials':len(noise),'all_expected_outcomes':True,'maximum_chi2':max(x[m]['chi2'] for x in noise for m in ['oracle','shortcut']),'maximum_parameter_error':max(x[m]['parameter_error'] for x in noise for m in ['oracle','shortcut']),'maximum_oracle_error':max(v for x in noise for v in x['oracle']['hidden'].values()),'minimum_shortcut_diagnostic_error':min(v for x in noise for k,v in x['shortcut']['hidden'].items() if k!='anchors'),'maximum_shortcut_anchor_error':max(x['shortcut']['hidden']['anchors'] for x in noise),'rows':noise}
    recoveries=[];shared=0.;sweep=[]
    for D in np.linspace(.8,1.2,41):
        y=O.predict_at(settings,float(D));rows=[{'input':e,'value':float(v),'sigma':sig} for e,v in zip(settings,y)];est=[mod.Model().fit(rows).diffusivity for mod in [O,S]]
        recoveries.append({'truth':float(D),'estimates':est,'maximum_error':max(abs(v-D) for v in est)});shared=max(shared,float(abs(y-S.predict_at(settings,D)).max()))
        sweep.append({'D':float(D),'groups':{k:{'physical':O.predict_at(v,float(D)).tolist(),'source':S.predict_at(v,float(D)).tolist(),'gap':error(S.predict_at(v,float(D)),O.predict_at(v,float(D)))} for k,v in groups.items()}})
    REPORT['identifiability']={'proof':'For two x-shear records with the same angle, coefficient = D+cos(angle)^2*(a^2+c^2/4)/(2D). Their difference is (C2-C1)/D, strictly decreasing for C2>C1 over the entire positive D interval. The included amplitude .6 and3 records at angle.15 have distinct C and globally identify D. The corresponding y-shear records provide the same independent relation. No zero-flow or directly known-D record is required.', 'maximum_shared_calibration_difference':shared,'maximum_noiseless_fit_error':max(r['maximum_error'] for r in recoveries),'recoveries':recoveries,'sampled_41_D_hidden_gaps':sweep,'minimum_sampled_diagnostic_gap':min(v['gap'] for r in sweep for k,v in r['groups'].items() if k!='anchors'),'qualification':'A sampled gap on fixed scored settings, not a uniform gap over the full control box; zero and shear settings intentionally agree.'}
    assert shared<1e-10 and REPORT['identifiability']['maximum_noiseless_fit_error']<1e-7 and REPORT['identifiability']['minimum_sampled_diagnostic_gap']>.1
    fields=[(R.experiment(a,b,c,d,.4),D,'corner') for D,a,b,c,d in product([.8,1.2],[-8.,8.],[-8.,8.],[-.8,.8],[-.8,.8])]
    random=np.random.default_rng(193087)
    fields += [(R.experiment(random.uniform(-8,8),random.uniform(-8,8),random.uniform(-.8,.8),random.uniform(-.8,.8),random.uniform(0,np.pi)),float(random.uniform(.8,1.2)),'interior') for _ in range(96)]
    fields += [(R.experiment(a,a,.1*a,-.1*a,.4),D,'weak_and_intermediate') for D,a in product([.8,1.,1.2],[0.,.1,.5,1.,2.,4.,6.,8.])]
    domain=[]
    for e,D,label in fields:
        K=tensor(O,e,D);fine=tensor(O,e,D,cutoff=28);source=tensor(S,e,D)
        reflected={**e,'a':-e['a'],'b':-e['b'],'c':-e['c'],'d':-e['d']};swapped={**e,'a':e['b'],'b':e['a'],'c':e['d'],'d':e['c']}
        first=(e['a']**2+e['c']**2/4)/2;second=(e['b']**2+e['d']**2/4)/2
        residual=max(abs(source[0,0]-D-first/source[1,1]),abs(source[1,1]-D-second/source[0,0]))
        row={'input':e,'D':D,'kind':label,'physical_tensor':K.tolist(),'source_tensor':source.tolist(),'signed_axis_gap':((np.diag(source)-np.diag(K))/np.diag(K)).tolist(),'spectral_refinement':float(abs(fine-K).max()),'minimum_enhancement_eigenvalue':float(np.linalg.eigvalsh(K-D*np.eye(2)).min()),'source_fixed_point_residual':float(residual),'source_minimum_eigenvalue':float(np.linalg.eigvalsh(source).min()),'flow_reversal_error':float(abs(tensor(O,reflected,D)-K).max()),'axis_exchange_error':float(abs(tensor(O,swapped,D)-K[::-1,::-1]).max())}
        if label=='corner' or len(domain)<64:
            ref=tensor(R,e,D);refined=tensor(R,e,D,coarse=96)
            row.update(reference_error=float(abs(ref-K).max()),reference_refinement=float(abs(refined-ref).max()),refined_reference_error=float(abs(refined-K).max()))
        domain.append(row)
    h=2*np.pi/48;minimum_rate=.8/h**2-8.8/(2*h)
    REPORT['domain']={'rows':domain,'corners':32,'interior':96,'weak_and_intermediate':24,'independent_reference_cases':sum('reference_error' in r for r in domain),'full_box_wellposedness':'D>=.8 uniformly elliptic on the periodic torus; smooth bounded divergence-free velocity preserves the uniform phase. The mean-zero cell equation is coercive in the gradient norm. Effective diffusivity is at least D times the identity. Source positive root gives Kx,Ky>=D for all nonnegative amplitude squares.','jump_positivity_proof':'For n>=48, h<=pi/24<2/11 and |u_i|<=8.8 throughout the box. Every rate >=h^-2(.8-4.4h)>0. Refinement increases this lower bound.','minimum_rate_lower_bound_n48':minimum_rate,'maximum_spectral_refinement':max(r['spectral_refinement'] for r in domain),'maximum_reference_error':max(r.get('reference_error',0) for r in domain),'maximum_refined_reference_error':max(r.get('refined_reference_error',0) for r in domain),'minimum_enhancement_eigenvalue':min(r['minimum_enhancement_eigenvalue'] for r in domain),'maximum_fixed_point_residual':max(r['source_fixed_point_residual'] for r in domain),'maximum_symmetry_error':max(max(r['flow_reversal_error'],r['axis_exchange_error']) for r in domain),'qualification':'Numerical cases cover all vertices and stated random/weak cases, not every interior field. All weak and sign-changing source differences are retained; no general source upper bound is claimed.'}
    assert minimum_rate>0 and REPORT['domain']['maximum_spectral_refinement']<1e-7
    assert REPORT['domain']['maximum_reference_error']<5e-4 and REPORT['domain']['maximum_refined_reference_error']<4e-5
    assert REPORT['domain']['minimum_enhancement_eigenvalue']>-1e-10 and REPORT['domain']['maximum_fixed_point_residual']<1e-10 and REPORT['domain']['maximum_symmetry_error']<1e-9
    temporal=[finite_time(R.experiment(a,a,0,0,0),D) for a,D in [(1.,1.),(4.,.8),(6.,1.2)]]
    count=counting_check(R.experiment(4,4,0,0,0),1.)
    REPORT['dynamic_checks']={'finite_time_displacement_moments':temporal,'counting_eigenvalue_curvature':count,'scope':'Exact moment propagation of the positive finite-grid random walk, not Monte Carlo and not a claim of equality at finite continuum time. Late-time slopes compare to the same-grid stationary moments; continuum errors are separately refined.'}
    assert max(r['relative_time_error'] for r in temporal)<5e-4 and count['relative_error']<1e-5
    REPORT['local_controls']=local_controls()
    REPORT['source_sha256']={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',BASE/'scripts/cellular_tracer_dispersion_baseline.py',Path(__file__)]}
    REPORT.update(status='science_and_local_controls_complete',seconds=time.monotonic()-start)
    (RESULTS/'cellular-tracer-dispersion-r1-validation.json').write_text(json.dumps(REPORT,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:REPORT[k] for k in ['status','actual_data','local_controls','seconds']},indent=2))
    print(json.dumps({k:v for k,v in REPORT['noise'].items() if k!='rows'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--generate',action='store_true')
    try:run(p.parse_args().generate)
    except Exception:
        REPORT.update(status='author_check_failed',traceback=traceback.format_exc())
        (RESULTS/f'author-check-failure-{time.time_ns()}.json').write_text(json.dumps(REPORT,indent=2,allow_nan=False)+'\n')
        raise
