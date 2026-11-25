"""Validate bottom-drag circulation feedback and the direct-PV-relaxation control."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import expm

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/rotating-layer'


def module(path):
    spec=importlib.util.spec_from_file_location('validation_'+str(abs(hash(str(path)))),path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def matrices(e,rate):
    k=e['wave'];f=e['rotation']
    correct=np.array([[0.,-.05*k,0.],[9.81*k,-rate,f],[0.,-f,-rate]])
    approximate=correct.copy();approximate[2,0]=rate*f/(.05*k)
    return correct,approximate


def checks(good,bad,ref):
    initial_errors=[];derivative_errors=[];independent=[];refinement=[];energy=[];pv_errors=[];shortcut_equivalence=[]
    all_hidden=sum(list(ref.hidden_inputs().values()),[])
    for es in ref.hidden_inputs().values():
        states=ref.full_states(es,.14)
        independent.append(float(max(abs(good.predict_at(es,.14)-states[:,0]))))
        refined=ref.full_states(es,.14,method='RK45',rtol=2e-12,max_step=.04)
        refinement.append(float(np.max(abs(states-refined))))
        for e,y in zip(es,states):
            correct,approx=matrices(e,.14);dy=correct@y;h,u,v=y
            physical_rate=.5*(9.81*h*dy[0]+.05*(u*dy[1]+v*dy[2]))
            loss=.5*.05*.14*(u*u+v*v)
            energy.append(abs(physical_rate+loss))
            pv_errors.append(abs(e['wave']*dy[2]-e['rotation']*dy[0]/.05+.14*e['wave']*v))
            y0=np.array([e['height'],e['along_velocity'],e['across_velocity']])
            shortcut_equivalence.append(abs(bad.predict_at([e],.14)[0]-(expm(approx*e['time'])@y0)[0]))
            zero=dict(e,time=0.)
            for source in [good,bad]:
                initial_errors.append(abs(source.predict_at([zero],.14)[0]-e['height']))
                dt=1e-6;derivative=(source.predict_at([dict(e,time=dt)],.14)[0]-e['height'])/dt
                derivative_errors.append(abs(derivative+.05*e['wave']*e['along_velocity']))
    inviscid=float(max(abs(good.predict_at(all_hidden,0.)-bad.predict_at(all_hidden,0.))))
    nonrotating=[dict(e,rotation=0.) for e in all_hidden]
    nonrotating_error=float(max(abs(good.predict_at(nonrotating,.14)-bad.predict_at(nonrotating,.14))))
    stability={'oracle':-np.inf,'shortcut':-np.inf}
    for rate in np.linspace(.08,.24,5):
        for f in np.linspace(-3,3,25):
            for k in [1,2,3]:
                e=ref.experiment(k,f,.0002,.001,.002,1.)
                for label,matrix in zip(stability,matrices(e,rate)):
                    stability[label]=max(stability[label],float(max(np.linalg.eigvals(matrix).real)))
    corners=[ref.experiment(k,f,h,u,v,t) for k in [1,3] for f in [-3.,3.] for h,u,v in [(.0004,.01,-.01),(-.0004,-.01,.01)] for t in [0.,.13,3.,40.]]
    corner_error=max(float(max(abs(good.predict_at(corners,rate)-ref.predict(corners,rate)))) for rate in [.08,.24])
    # The closure is stable, but does not inherit mechanical energy dissipation.
    e=ref.experiment(1,2.,.0002,0.,.004,0.);state=np.array([e['height'],0.,e['across_velocity']])
    correct,approx=matrices(e,.14);metric=np.diag([9.81,.05,.05])
    counterexample=float(.5*state@metric@approx@state)
    correct_rate=float(.5*state@metric@correct@state)
    assert max(independent)<1e-10 and max(refinement)<1e-10 and corner_error<1e-10
    assert max(initial_errors)<1e-16 and max(derivative_errors)<1e-8
    assert max(energy)<1e-16 and max(pv_errors)<1e-14 and max(shortcut_equivalence)<1e-15
    assert inviscid<1e-14 and nonrotating_error==0 and max(stability.values())<0
    assert counterexample>0 and correct_rate<0
    return {'oracle_momentum_reference_max_m':max(independent),'reference_dop853_vs_rk45_max_state_error':max(refinement),
            'initial_height_error_both_controls_m':max(initial_errors),'initial_height_derivative_error_both_controls_m_per_s':max(derivative_errors),
            'source_analytic_vs_independent_approximate_generator_m':max(shortcut_equivalence),
            'mechanical_energy_balance_error_per_density':max(energy),'drag_curl_PV_balance_error':max(pv_errors),
            'inviscid_equivalence_m':inviscid,'nonrotating_equivalence_m':nonrotating_error,
            'maximum_generator_real_eigenvalue_domain':stability,'domain_corner_reference_error_m':corner_error,
            'shortcut_energy_defect':{'input':e,'physical_energy_rate_per_density':correct_rate,
                'shortcut_energy_rate_per_density':counterexample,
                'interpretation':'Direct PV relaxation adds height-dependent transverse forcing. This stable approximation can inject mechanical energy; it is a consequence of its incorrect dissipative law, not a coding defect.'}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true');parser.add_argument('--noise-trials',type=int,default=256)
    args=parser.parse_args();start=time.time()
    good=module(TASK/'solution/model.py');bad=module(ROOT/'scripts/rotating_layer_baseline.py');ref=module(TASK/'tests/reference.py')
    metadata=json.loads((TASK/'tests/metadata.json').read_text());true=ref.TRUE_PARAMETER;sigma=metadata['measurement_sigma'];limit=metadata['prediction_limit']
    inputs=ref.calibration_inputs();exact=ref.predict(inputs,true)
    if args.generate:
        values=exact+np.random.default_rng(metadata['calibration_seed']).normal(0,sigma,len(inputs))
        records=[dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,values)]
        for directory in ['environment','tests']:(TASK/directory/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records==json.loads((TASK/'tests/data/calibration.json').read_text()) and [r['input'] for r in records]==inputs
    assert all(r['sigma']==sigma for r in records)
    hidden=ref.hidden_inputs();truth={name:ref.predict(es,true) for name,es in hidden.items()}
    def chi(model,rs):
        residual=(model.predict(inputs)-np.array([r['value'] for r in rs]))/sigma
        return float(residual@residual/(len(rs)-1))
    def scores(model):return {name:float(np.sqrt(np.mean((model.predict(es)-truth[name])**2))) for name,es in hidden.items()}
    report={'revision':1,'calibration_seed':metadata['calibration_seed'],'noise_seed':metadata['noise_seed'],'noise_trials':args.noise_trials,'controls':{}}
    for label,source in [('oracle',good),('shortcut',bad)]:
        model=source.Model().fit(records);result={'parameter':model.drag_rate,'parameter_relative_error':abs(model.drag_rate/true-1),
            'calibration_chi2':chi(model,records),'hidden':scores(model)}
        assert result['calibration_chi2']<1.5 and result['parameter_relative_error']<.03
        assert (max(result['hidden'].values())<limit if label=='oracle' else min(result['hidden'].values())>limit)
        report['controls'][label]=result
    rng=np.random.default_rng(metadata['noise_seed']);parameters=[];chi2s=[]
    for _ in range(args.noise_trials):
        sample=[dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,exact+rng.normal(0,sigma,len(inputs)))]
        a=good.Model().fit(sample);b=bad.Model().fit(sample);assert a.drag_rate==b.drag_rate
        parameters.append(a.drag_rate);chi2s.append(chi(a,sample))
    assert max(chi2s)<1.5 and max(abs(np.array(parameters)/true-1))<.03
    report['noise']={'parameter_min':min(parameters),'parameter_max':max(parameters),'parameter_relative_error_max':float(max(abs(np.array(parameters)/true-1))),
        'calibration_chi2_max':max(chi2s),'calibration_pass_fraction':1.}
    extrema={}
    for label,source in [('oracle',good),('shortcut',bad)]:
        errors=[]
        for rate in [min(parameters),max(parameters)]:
            model=source.Model();model.drag_rate=rate;errors.extend(scores(model).values())
        extrema[label]={'min':min(errors),'max':max(errors)}
    assert extrema['oracle']['max']<limit and extrema['shortcut']['min']>limit
    report['hidden_parameter_extrema_check']=extrema;report['physical_checks']=checks(good,bad,ref);report['seconds']=time.time()-start
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TASK/'instruction.md',TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',TASK/'tests/metadata.json',TASK/'environment/data/calibration.json',ROOT/'scripts/rotating_layer_baseline.py',Path(__file__)]}
    output=ROOT/'jobs/rotating-layer-validation';output.mkdir(parents=True,exist_ok=True);(output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('rotating-layer PASS',output/'summary.json',report['seconds'])


if __name__=='__main__':main()
