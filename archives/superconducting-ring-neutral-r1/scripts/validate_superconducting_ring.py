"""Scientific controls for a fixed-winding, fixed-volume London ring."""
import argparse,hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
from scipy.integrate import quad
ROOT=Path(__file__).resolve().parents[1];TASK=ROOT/'tasks/superconducting-ring'


def module(path):
    spec=importlib.util.spec_from_file_location('check_'+str(abs(hash(str(path)))),path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--generate',action='store_true');args=parser.parse_args();started=time.time()
    good=module(TASK/'solution/model.py');bad=module(ROOT/'scripts/superconducting_ring_baseline.py');ref=module(TASK/'tests/reference.py')
    meta=json.loads((TASK/'tests/metadata.json').read_text());true=ref.TRUE_PARAMETER;inputs=ref.calibration_inputs();exact=ref.predict(inputs,true);sigma=meta['measurement_sigma']
    def records_at(values):return [dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,values)]
    if args.generate:
        records=records_at(exact+np.random.default_rng(meta['calibration_seed']).normal(0,sigma,len(inputs)))
        for directory in ['environment','tests']:(TASK/directory/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text());assert records==json.loads((TASK/'tests/data/calibration.json').read_text())
    hidden=ref.hidden_inputs();truth={name:ref.predict(es,true) for name,es in hidden.items()}
    def errors(model):return {name:float(np.linalg.norm(model.predict(es)-truth[name])/np.linalg.norm(truth[name])) for name,es in hidden.items()}
    def scores(model,sample):
        residual=(model.predict(inputs)-np.array([r['value'] for r in sample]))/sigma
        return dict(parameter=model.kinetic_scale,parameter_relative_error=abs(model.kinetic_scale/true-1),calibration_chi2=float(residual@residual)/(len(inputs)-1),hidden=errors(model))
    controls={}
    for label,source in [('oracle',good),('shortcut',bad)]:
        result=scores(source.Model().fit(records),records);controls[label]=result
        assert result['parameter_relative_error']<.03 and result['calibration_chi2']<1.5
        assert (max(result['hidden'].values())<.04 if label=='oracle' else min(result['hidden'].values())>.04),result
    rng=np.random.default_rng(meta['noise_seed']);parameters=[];chi=[];oe=[];be=[]
    for _ in range(256):
        sample=records_at(exact+rng.normal(0,sigma,len(inputs)));a=good.Model().fit(sample);b=bad.Model().fit(sample)
        assert abs(a.kinetic_scale-b.kinetic_scale)<1e-10
        row=scores(a,sample);parameters.append(a.kinetic_scale);chi.append(row['calibration_chi2']);oe.extend(row['hidden'].values());be.extend(errors(b).values())
    assert max(abs(np.array(parameters)/true-1))<.03 and np.mean(np.array(chi)<1.5)>=.99
    assert max(oe)<.04 and min(be)>.04
    agreement=[];constraint=[];work=[];volume=[];baseline_cycle=[]
    for scale in [4.,9.,16.]:
        es=[ref.experiment(float(rng.uniform(.7,1.6)),float(rng.uniform(-.35,.35)),int(rng.integers(-3,4)),'radial_force') for _ in range(50)]
        agreement.append(float(np.max(abs(ref.predict(es,scale)-good.predict_at(es,scale)))))
        for e in es:
            current=good.predict_at([{**e,'observable':'current'}],scale)[0];r=e['radius']
            constraint.append(abs(e['winding']-np.pi*r*r*e['magnetic_field']-ref.inductance(r,scale)*current))
            volume.append(abs((.012/np.sqrt(r))**2*r-.012**2))
        for n in [-2,0,3]:
            a,b=.75,1.5;b1,b2=-.2,.3
            def cycle(source):
                radial=lambda r,bias:source.predict_at([ref.experiment(r,bias,n,'radial_force')],scale)[0]
                magnetic=lambda bias,r:np.pi*r*r*source.predict_at([ref.experiment(r,bias,n)],scale)[0]
                return (quad(lambda r:radial(r,b1),a,b,epsabs=1e-12)[0]+quad(lambda bias:magnetic(bias,b),b1,b2,epsabs=1e-12)[0]
                        +quad(lambda r:radial(r,b2),b,a,epsabs=1e-12)[0]+quad(lambda bias:magnetic(bias,a),b2,b1,epsabs=1e-12)[0])
            work.append(abs(cycle(good)));baseline_cycle.append(abs(cycle(bad)))
    assert max(agreement)<1e-12 and max(constraint)<1e-12 and max(work)<1e-11 and max(volume)<1e-18
    report=dict(revision=1,controls=controls,noise=dict(trials=256,parameter_range=[min(parameters),max(parameters)],relative_parameter_error_max=float(max(abs(np.array(parameters)/true-1))),calibration_pass_fraction=float(np.mean(np.array(chi)<1.5)),calibration_chi2_max=max(chi),oracle_hidden_error_max=max(oe),shortcut_hidden_error_min=min(be)),physical_checks=dict(virtual_work_force_error_max=max(agreement),fluxoid_residual_max=max(constraint),fixed_volume_residual_max=max(volume),closed_cycle_energy_error_max=max(work),shortcut_closed_cycle_error_max=max(baseline_cycle)),measurement_sigma=sigma,prediction_limit=.04,seconds=time.time()-started)
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TASK/'instruction.md',TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',ROOT/'scripts/superconducting_ring_baseline.py']}
    (ROOT/'results/superconducting-ring-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
