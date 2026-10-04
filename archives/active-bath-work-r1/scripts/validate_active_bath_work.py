"""Independent finite-mass controls for active-bath-work."""
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
from scipy.linalg import solve_continuous_lyapunov

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/active-bath-work'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


good=load(TASK/'solution/model.py','good');bad=load(ROOT/'scripts/active_bath_work_baseline.py','bad');ref=load(TASK/'tests/reference.py','reference')


def run(count):
    records=json.loads((TASK/'tests/data/calibration.json').read_text());inputs=[r['input'] for r in records];sigma=np.array([r['sigma'] for r in records]);clean=ref.predict(inputs,.8)
    hidden=ref.hidden_inputs();truth={key:ref.predict(es,.8) for key,es in hidden.items()}
    def scores(module,d):return {key:float(np.sqrt(np.mean((module.predict_at(es,d)-truth[key])**2)/np.mean(truth[key]**2))) for key,es in hidden.items()}
    controls={}
    for label,module in [('oracle',good),('shortcut',bad)]:
        model=module.Model().fit(records);res=(model.predict(inputs)-np.array([r['value'] for r in records]))/sigma
        controls[label]=dict(activity=model.activity,calibration_chi2=float(res@res/(len(records)-1)),hidden=scores(module,model.activity))
    rng=np.random.default_rng(934173);samples=[]
    for _ in range(count):
        values=clean+rng.normal(size=len(inputs))*sigma;runs=[dict(input=e,value=float(v),sigma=float(s)) for e,v,s in zip(inputs,values,sigma)]
        a=good.Model().fit(runs);b=bad.Model().fit(runs);assert a.activity==b.activity
        res=(a.predict(inputs)-values)/sigma
        samples.append(dict(activity=a.activity,chi2=float(res@res/(len(inputs)-1)),oracle=scores(good,a.activity),shortcut=scores(bad,b.activity)))
    # Independent finite-mass reference refinement at all scored preparations.
    all_inputs=inputs+sum(hidden.values(),[])
    comparison=max(float(max(abs(good.predict_at(all_inputs,d)-ref.predict(all_inputs,d)))) for d in [.4,.8,1.2])
    refinement=float(max(abs(ref.predict(all_inputs,.8)-ref.predict(all_inputs,.8,.5))))
    exact_cal=max(float(max(abs(good.predict_at(inputs,d)-bad.predict_at(inputs,d)))) for d in [.4,.8,1.2])
    corner_error=0.;corner_refinement=0.;positive=1.;balance=0.;fvariance=0.;direct_scaling=0.;reflection=0.
    for d in [.4,1.2]:
        for k in [.6,1.6]:
            for b,chi,alpha in [(-3.,3.,.2),(3.,-3.,3.),(3.,3.,3.),(-3.,-3.,.2),(0.,0.,3.)]:
                es=[ref.experiment(k,b,chi,alpha,readout,lag) for readout,lag in [('work',None),('xx',0.),('xy',.2),('yy',3.)]]
                a=good.predict_at(es,d);v=ref.predict(es,d);fine=ref.predict(es,d,.5)
                corner_error=max(corner_error,float(max(abs(a-v))));corner_refinement=max(corner_refinement,float(max(abs(v-fine))))
                m=.0008;A,C=ref.finite_state(d,k,b,chi,alpha,m)
                positive=min(positive,float(min(np.linalg.eigvalsh(C))))
                # m times stationary mechanical energy balance: active work
                # equals drag dissipation minus the thermal-noise injection.
                balance=max(balance,abs(np.trace(C[2:4,4:])-np.trace(C[2:4,2:4])+1.4))
                fvariance=max(fvariance,float(np.max(abs(C[4:,4:]-d/alpha*np.eye(2)))))
                # Build the unscaled physical6D process independently to verify
                # every mass/noise normalization of the reference variables.
                physical=np.zeros((6,6));physical[:2,2:4]=np.eye(2);physical[2:4,:2]=-k/m*np.eye(2);physical[2:4,2:4]=(-np.eye(2)+b*ref.J)/m;physical[2:4,4:]=np.eye(2)/m;physical[4:,4:]=(-np.eye(2)+chi*ref.J)/(alpha*m)
                diffusion=np.zeros((6,6));diffusion[2:4,2:4]=1.4/m**2*np.eye(2);diffusion[4:,4:]=2*d/(alpha*m)**2*np.eye(2);unscaled=solve_continuous_lyapunov(physical,-diffusion)
                direct_scaling=max(direct_scaling,abs(m*np.trace(unscaled[2:4,4:])-np.trace(C[2:4,4:])))
                reversed_es=[dict(e,field=-b,chirality=-chi) for e in es];reverse=good.predict_at(reversed_es,d);sign=np.array([1.,1.,-1.,1.]);reflection=max(reflection,float(max(abs(a-sign*reverse))))
    # Analytic checks independent of the matrix reference: B=chi=0 yields
    # the scalar force/velocity relaxation competition; slow force gives
    # the instantaneous mobility result after multiplying by alpha.
    scalar=max(abs(good.predict_at([ref.experiment(1.,0.,0.,alpha)],.8)[0]-1.6/(alpha+1)) for alpha in [.2,.7,3.])
    slow=[]
    for alpha in [10.,100.,1000.]:
        e=ref.experiment(1.,1.3,.7,alpha)
        slow.append(abs(alpha*(good.predict_at([e],.8)[0]-bad.predict_at([e],.8)[0])))
    assert slow[-1]<slow[0]
    local={}
    for name,source in [('oracle',TASK/'solution/model.py'),('shortcut',ROOT/'scripts/active_bath_work_baseline.py')]:
        with tempfile.TemporaryDirectory(prefix='active-work-'+name+'-') as directory:
            r=Path(directory);shutil.copytree(TASK/'environment',r/'app');shutil.copytree(TASK/'tests',r/'tests');shutil.copy2(source,r/'app/model.py')
            p=subprocess.run([sys.executable,'-m','pytest','-q',str(r/'app/test_public.py'),str(r/'tests/test_hidden.py')],cwd=r/'app',env=dict(os.environ,PYTHONPATH=str(r/'app'),OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True)
            local[name]=dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    report=dict(revision=1,noise_trials=count,calibration_seed=934171,noise_seed=934173,controls=controls,
        all_scored_reference_error=comparison,all_scored_mass_refinement=refinement,calibration_equivalence=exact_cal,
        corner_reference_error=corner_error,corner_mass_refinement=corner_refinement,minimum_finite_covariance_eigenvalue=positive,
        finite_mechanical_energy_balance=balance,scaled_force_variance_error=fvariance,direct_physical_scaling_error=direct_scaling,
        simultaneous_field_chirality_reflection=reflection,scalar_relaxation_limit=scalar,slow_force_limit_errors=slow,
        noise=dict(calibration_passes=sum(s['chi2']<1.5 for s in samples),parameter_passes=sum(abs(s['activity']/.8-1)<.03 for s in samples),
            oracle_passes=sum(max(s['oracle'].values())<.03 for s in samples),shortcut_passes=sum(max(s['shortcut'].values())<.03 for s in samples),
            maximum_oracle_error=max(max(s['oracle'].values()) for s in samples),minimum_shortcut_error=min(min(s['shortcut'].values()) for s in samples),
            maximum_chi2=max(s['chi2'] for s in samples),activity_range=[min(s['activity'] for s in samples),max(s['activity'] for s in samples)]),
        local_pytest=local,noise_realizations=samples)
    assert comparison<1e-6 and refinement<1e-6 and corner_error<1e-6 and corner_refinement<1e-6
    assert exact_cal<1e-12 and positive>0 and balance<1e-9 and fvariance<1e-9 and direct_scaling<1e-8 and reflection<1e-12
    assert all(report['noise'][k]==count for k in ['calibration_passes','parameter_passes','oracle_passes'])
    assert report['noise']['shortcut_passes']==0 and report['noise']['minimum_shortcut_error']>.03
    assert local['oracle']['returncode']==0 and local['shortcut']['returncode']==1
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');parser.add_argument('--noise-trials',type=int,default=256);args=parser.parse_args()
    if args.generate:
        inputs=ref.calibration_inputs();values=ref.predict(inputs,.8);rng=np.random.default_rng(934171)
        records=[dict(input=e,value=float(v+.001*rng.normal()),sigma=.001) for e,v in zip(inputs,values)]
        for area in ['environment','tests']:(TASK/area/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
    report=run(args.noise_trials);path=ROOT/'results/active-bath-work-validation.json';path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['noise_realizations','local_pytest']},indent=2))
