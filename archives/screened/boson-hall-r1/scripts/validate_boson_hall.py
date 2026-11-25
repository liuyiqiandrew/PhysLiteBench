"""Validate boson heat transport with independent Bloch loops and edge currents."""
import argparse, hashlib, importlib.util, json, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];TASK=ROOT/'tasks/boson-hall'
def module(path):
    spec=importlib.util.spec_from_file_location('validation_'+str(abs(hash(str(path)))),path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

def physical_checks(good,bad,ref):
    hidden=sum(ref.hidden_inputs().values(),[]);cal=ref.calibration_inputs()
    truth=ref.predict(hidden);prediction=good.predict_at(hidden,ref.TRUE_PARAMETER)
    agreement=float(max(abs(prediction-truth))/max(abs(truth)))
    refinement=float(max(abs(ref.predict(hidden,ref.TRUE_PARAMETER)-np.array([ref.thermal(e['mass'],e['handedness'],e['temperature'],ref.TRUE_PARAMETER,256) for e in hidden])))/max(abs(truth)))
    corner=[];signs=[];chern=[];population=[];derivative=[]
    for mass in [-1.6,-1.2,-.6]:
        for handedness in [-1,1]:
            energy,curvature=ref.plaquettes(mass,handedness,192)
            chern.append((np.mean(curvature,axis=(0,1))*2*np.pi).tolist())
            assert np.max(abs(np.abs(chern[-1])-1))<1e-12
            assert np.max(abs(np.sum(curvature,axis=-1)))<1e-10
            population.append(float(energy.min()))
            # The analytic d dot (dx d cross dy d) is a third curvature construction.
            n=96;k=-np.pi+2*np.pi*(np.arange(n)+.5)/n;x,y=np.meshgrid(k,k,indexing='ij')
            d=np.stack([np.sin(x),handedness*np.sin(y),mass+np.cos(x)+np.cos(y)],axis=-1)
            dx=np.stack([np.cos(x),np.zeros_like(x),-np.sin(x)],axis=-1)
            dy=np.stack([np.zeros_like(y),handedness*np.cos(y),-np.sin(y)],axis=-1)
            lower=np.sum(d*np.cross(dx,dy),axis=-1)/(2*np.linalg.norm(d,axis=-1)**3)
            _,response,_=good.band_data(mass,handedness,n)
            derivative.append(float(max(abs(response[...,0]+lower).ravel())))
            for scale in [.8,1.2]:
                for temperature in [.35,1.5]:
                    e=ref.experiment(mass,handedness,temperature)
                    exact=ref.predict([e],scale)[0];pred=good.predict_at([e],scale)[0]
                    corner.append(float(abs(exact-pred)/max(abs(exact),1e-6)))
                    reverse=good.predict_at([dict(e,handedness=-handedness)],scale)[0]
                    signs.append(abs(pred+reverse))
    assert min(population)>0 and max(corner)<1e-4 and max(signs)<1e-12 and max(derivative)<1e-12
    assert agreement<1e-5 and refinement<1e-5
    identical=float(max(abs(good.predict_at(cal,1.04)-bad.predict_at(cal,1.04))))
    assert identical==0
    fits=[]
    for scale in [.8,1.04,1.2]:
        values=ref.predict(cal,scale);records=[dict(input=e,value=float(y),sigma=.002) for e,y in zip(cal,values)]
        fits.append(abs(good.Model().fit(records).exchange_scale-scale))
    assert max(fits)<1e-12
    return dict(oracle_reference_relative_error=agreement,reference_refinement_relative_error=refinement,
        corner_relative_error_max=max(corner),handedness_reversal_error_max=max(signs),chern_numbers=chern,
        minimum_band_energy=min(population),analytic_velocity_curvature_error_max=max(derivative),
        calibration_closure_difference=identical,noiseless_fit_errors=fits)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--generate',action='store_true');parser.add_argument('--noise-trials',type=int,default=256);args=parser.parse_args();start=time.time()
    good=module(TASK/'solution/model.py');bad=module(ROOT/'scripts/boson_hall_baseline.py');ref=module(TASK/'tests/reference.py')
    metadata=json.loads((TASK/'tests/metadata.json').read_text());true=ref.TRUE_PARAMETER;sigma=metadata['measurement_sigma'];limit=metadata['prediction_limit']
    inputs=ref.calibration_inputs();exact=ref.predict(inputs,true)
    if args.generate:
        values=exact+np.random.default_rng(metadata['calibration_seed']).normal(0,sigma,len(inputs))
        records=[dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,values)]
        for directory in ['environment','tests']:
            (TASK/directory/'data').mkdir(exist_ok=True);(TASK/directory/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text());assert records==json.loads((TASK/'tests/data/calibration.json').read_text())
    assert [r['input'] for r in records]==inputs and all(r['sigma']==sigma for r in records)
    hidden=ref.hidden_inputs();truth={key:ref.predict(es,true) for key,es in hidden.items()}
    def chi(model,rs):
        residual=(model.predict(inputs)-np.array([r['value'] for r in rs]))/sigma
        return float(residual@residual/(len(rs)-1))
    def scores(model):return {key:float(np.sqrt(np.mean((model.predict(es)-truth[key])**2)/np.mean(truth[key]**2))) for key,es in hidden.items()}
    report=dict(revision=1,calibration_seed=metadata['calibration_seed'],noise_seed=metadata['noise_seed'],noise_trials=args.noise_trials,controls={})
    for name,source in [('oracle',good),('shortcut',bad)]:
        model=source.Model().fit(records);result=dict(parameter=model.exchange_scale,parameter_relative_error=abs(model.exchange_scale/true-1),calibration_chi2=chi(model,records),hidden=scores(model))
        assert result['calibration_chi2']<1.5 and result['parameter_relative_error']<.03
        assert max(result['hidden'].values())<limit if name=='oracle' else min(result['hidden'].values())>limit
        report['controls'][name]=result
    rng=np.random.default_rng(metadata['noise_seed']);parameters=[];chi2s=[];noise_good=[];noise_bad=[]
    for _ in range(args.noise_trials):
        sample=[dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,exact+rng.normal(0,sigma,len(inputs)))]
        a=good.Model().fit(sample);b=bad.Model().fit(sample);assert a.exchange_scale==b.exchange_scale
        parameters.append(a.exchange_scale);chi2s.append(chi(a,sample));noise_good.append(max(scores(a).values()));noise_bad.append(min(scores(b).values()))
        assert noise_good[-1]<limit and noise_bad[-1]>limit
    assert max(chi2s)<1.5 and max(abs(np.array(parameters)/true-1))<.03
    report['noise']=dict(parameter_min=min(parameters),parameter_max=max(parameters),calibration_chi2_max=max(chi2s),oracle_hidden_passes=args.noise_trials,shortcut_hidden_passes=0,oracle_hidden_max=max(noise_good),shortcut_hidden_min=min(noise_bad))
    report['physical_checks']=physical_checks(good,bad,ref);report['seconds']=time.time()-start
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TASK/'instruction.md',TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',TASK/'tests/metadata.json',TASK/'environment/data/calibration.json',ROOT/'scripts/boson_hall_baseline.py',Path(__file__)]}
    output=ROOT/'results/boson-hall-validation.json';output.write_text(json.dumps(report,indent=2)+'\n');print('boson-hall PASS',output,report['seconds']);print(json.dumps(report['controls'],indent=2));print(json.dumps(report['physical_checks'],indent=2))

if __name__=='__main__':main()
