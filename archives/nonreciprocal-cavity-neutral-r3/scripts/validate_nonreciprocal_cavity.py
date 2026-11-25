"""Scientific controls for modal thermal emission from a passive biased cavity."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/nonreciprocal-cavity'


def module(path):
    spec=importlib.util.spec_from_file_location('validation_'+str(abs(hash(str(path)))),path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def metrics(model,records,true):
    prediction=model.predict([r['input'] for r in records])
    residual=(prediction-np.array([r['value'] for r in records]))/np.array([r['sigma'] for r in records])
    return {'parameter':model.loss_rate,'parameter_relative_error':abs(model.loss_rate/true-1),
            'calibration_chi2':float(residual@residual)/(len(records)-1)}


def physical_checks(good,bad,ref):
    defects=[];trace_errors=[];reference_errors=[];equilibrium=[];reversal=[];reciprocity=[]
    eigenvalues=[];unitarity=[];shortcut_equilibrium=[]
    for rate in [0.,.1,.27,.6]:
        for phase in [-np.pi,-1.7,0.,.8,np.pi]:
            for frequency in [-2.,-.37,.6,2.]:
                e=ref.experiment(frequency,phase,.7,(.7,.7,.7))
                s=good.scattering_matrices([e],rate)[0]
                emission=np.eye(3)-s@s.conj().T
                absorption=np.eye(3)-s.conj().T@s
                _,b=ref.bath_transfer(e,rate)
                defects.append(float(np.max(abs(emission-b@b.conj().T))))
                trace_errors.append(float(abs(np.trace(emission-absorption))))
                eigenvalues.extend(np.linalg.eigvalsh(emission).tolist())
                eigenvalues.extend(np.linalg.eigvalsh(absorption).tolist())
                equilibrium.append(float(np.max(abs(ref.covariance(e,rate)-.7*np.eye(3)))))
                es=[dict(e,output_port=p) for p in [-1,0,1,2]]
                reference_errors.append(float(max(abs(good.predict_at(es,rate)-ref.predict(es,rate)))))
                shortcut_equilibrium.append(float(max(abs(bad.predict_at(es[1:],rate)-.7))))
                reverse=good.scattering_matrices([dict(e,flux_phase=-phase)],rate)[0]
                reversal.append(float(np.max(abs(emission-np.conj(np.eye(3)-reverse.conj().T@reverse)))))
                if phase in [-np.pi,0.,np.pi]:
                    reciprocity.append(float(max(abs(np.diag(emission-absorption)))))
                if rate==0:
                    unitarity.append(float(np.max(abs(emission))))
    assert max(defects)<1e-12 and max(trace_errors)<1e-12 and max(reference_errors)<1e-12
    assert min(eigenvalues)>-1e-12 and max(eigenvalues)<1+1e-12
    assert max(equilibrium)<1e-12 and max(reversal)<1e-12 and max(reciprocity)<1e-12 and max(unitarity)<1e-12
    assert max(shortcut_equilibrium)<1e-12
    # A profile over the full permitted interval checks that the total-output
    # calibration identifies the loss scale, rather than a reciprocal alias.
    inputs=ref.calibration_inputs();exact=ref.predict(inputs,ref.TRUE_PARAMETER)
    rates=np.linspace(.1,.6,1001)
    costs=np.array([np.sum((good.predict_at(inputs,r)-exact)**2) for r in rates])
    local_minima=np.flatnonzero((costs[1:-1]<costs[:-2])&(costs[1:-1]<costs[2:]))+1
    assert len(local_minima)==1 and abs(rates[local_minima[0]]-ref.TRUE_PARAMETER)<1e-12
    fitted=[]
    for rate in [.1001,.17,.35,.49,.5999]:
        records=[dict(input=e,value=float(y),sigma=.01) for e,y in zip(inputs,ref.predict(inputs,rate))]
        fitted.append(abs(good.Model().fit(records).loss_rate-rate))
    assert max(fitted)<1e-7
    return {'langevin_emission_matrix_identity_error':max(defects),'total_emission_absorption_trace_error':max(trace_errors),
            'independent_reference_corner_error':max(reference_errors),'defect_eigenvalue_range':[min(eigenvalues),max(eigenvalues)],
            'common_temperature_output_covariance_error':max(equilibrium),'bias_reversal_identity_error':max(reversal),
            'reciprocal_port_emission_absorption_error':max(reciprocity),'lossless_unitarity_error':max(unitarity),
            'shortcut_equilibrium_port_error':max(shortcut_equilibrium),'noiseless_parameter_recovery_max_error':max(fitted),
            'full_interval_calibration_profile_minima':len(local_minima)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true');parser.add_argument('--noise-trials',type=int,default=256)
    parser.add_argument('--output',type=Path,default=ROOT/'results/nonreciprocal-cavity-r3-validation.json')
    args=parser.parse_args();started=time.time()
    good=module(TASK/'solution/model.py');bad=module(ROOT/'scripts/nonreciprocal_cavity_baseline.py');ref=module(TASK/'tests/reference.py')
    metadata=json.loads((TASK/'tests/metadata.json').read_text());true=ref.TRUE_PARAMETER
    inputs=ref.calibration_inputs();exact=ref.predict(inputs,true)
    sigma=metadata['measurement_control_fraction']*np.array([np.mean(e['body_occupation']) for e in inputs])
    def records_at(values):return [dict(input=e,value=float(y),sigma=float(s)) for e,y,s in zip(inputs,values,sigma)]
    if args.generate:
        records=records_at(exact+np.random.default_rng(metadata['calibration_seed']).normal(size=len(inputs))*sigma)
        for directory in ['environment','tests']:(TASK/directory/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records==json.loads((TASK/'tests/data/calibration.json').read_text())
    assert [r['input'] for r in records]==inputs
    hidden=ref.hidden_inputs();truth={name:ref.predict(es,true) for name,es in hidden.items()}
    def scores(model):return {name:float(np.sqrt(np.mean((model.predict(es)-truth[name])**2))) for name,es in hidden.items()}
    assert np.array_equal(np.array([r['sigma'] for r in records]),sigma)
    report={'revision':metadata['revision'],
            'measurement_sigma_definition':metadata['measurement_sigma_definition'],
            'measurement_sigma_range':[float(sigma.min()),float(sigma.max())],'calibration_seed':metadata['calibration_seed'],'noise_seed':metadata['noise_seed'],
            'noise_trials':args.noise_trials,'prediction_limit':metadata['prediction_limit'],'controls':{}}
    for label,source in [('oracle',good),('shortcut',bad)]:
        m=source.Model().fit(records);result=metrics(m,records,true);result['hidden']=scores(m)
        result['calibration_reference_max_error']=float(max(abs(source.predict_at(inputs,true)-exact)))
        assert result['calibration_chi2']<1.5 and result['parameter_relative_error']<.03
        assert result['calibration_reference_max_error']<1e-12
        assert (max(result['hidden'].values())<metadata['prediction_limit'] if label=='oracle' else min(result['hidden'].values())>metadata['prediction_limit'])
        report['controls'][label]=result
    rng=np.random.default_rng(metadata['noise_seed']);parameters=[];chi2s=[];noise_errors={'oracle':[],'shortcut':[]}
    for _ in range(args.noise_trials):
        sample=records_at(exact+rng.normal(size=len(inputs))*sigma)
        a=good.Model().fit(sample);b=bad.Model().fit(sample);result=metrics(a,sample,true)
        assert abs(a.loss_rate-b.loss_rate)<1e-10
        parameters.append(a.loss_rate);chi2s.append(result['calibration_chi2'])
        noise_errors['oracle'].extend(scores(a).values());noise_errors['shortcut'].extend(scores(b).values())
    report['noise']={'parameter_min':min(parameters),'parameter_max':max(parameters),
                     'parameter_relative_error_max':float(max(abs(np.array(parameters)/true-1))),
                     'calibration_chi2_max':max(chi2s),'calibration_pass_fraction':float(np.mean(np.array(chi2s)<1.5)),
                     'oracle_hidden_error_max':max(noise_errors['oracle']),'shortcut_hidden_error_min':min(noise_errors['shortcut'])}
    assert report['noise']['parameter_relative_error_max']<.03 and report['noise']['calibration_pass_fraction']>=.99
    assert max(noise_errors['oracle'])<metadata['prediction_limit'] and min(noise_errors['shortcut'])>metadata['prediction_limit']
    report['physical_checks']=physical_checks(good,bad,ref)
    report['seconds']=time.time()-started
    files=[TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py']
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('nonreciprocal-cavity PASS',args.output,report['seconds'])


if __name__=='__main__':main()
