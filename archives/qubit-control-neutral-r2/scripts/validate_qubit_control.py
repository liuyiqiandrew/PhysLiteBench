"""Controls for two probes coupled to a quantum resonator bath."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/qubit-control'


def module(path):
    spec=importlib.util.spec_from_file_location('check_'+str(abs(hash(str(path)))),path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true')
    parser.add_argument('--noise-trials',type=int,default=256)
    parser.add_argument('--output',type=Path,default=ROOT/'results/qubit-neutral-r2-validation.json')
    args=parser.parse_args();started=time.time()
    good=module(TASK/'solution/model.py');bad=module(ROOT/'scripts/qubit_control_baseline.py');ref=module(TASK/'tests/reference.py')
    meta=json.loads((TASK/'tests/metadata.json').read_text());true=ref.TRUE_PARAMETER
    inputs=ref.calibration_inputs();exact=ref.predict(inputs,true);sigma=meta['measurement_sigma']
    def records_at(values):
        return [dict(experiment=e,probability=float(v),sigma=sigma) for e,v in zip(inputs,values)]
    if args.generate:
        records=records_at(exact+np.random.default_rng(meta['calibration_seed']).normal(0,sigma,len(inputs)))
        for directory in ['environment','tests']:
            (TASK/directory/'data/calibration.json').write_text(json.dumps(records,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records==json.loads((TASK/'tests/data/calibration.json').read_text())
    hidden=ref.hidden_inputs();truth={n:ref.predict(es,true) for n,es in hidden.items()}
    terms={n:good.signal_terms(es) for n,es in hidden.items()};bad_terms={n:bad.signal_terms(es) for n,es in hidden.items()}
    def errors(parameter,correct):
        fn,t=(good.signal,terms) if correct else (bad.signal,bad_terms)
        return {n:float(np.sqrt(np.mean((fn(t[n],parameter)-truth[n])**2))) for n in hidden}
    def scores(model,sample,correct):
        residual=(model.predict(inputs)-np.array([r['probability'] for r in sample]))/sigma
        return dict(gamma=model.gamma,parameter_relative_error=abs(model.gamma/true-1),calibration_chi2=float(residual@residual)/(len(inputs)-1),hidden=errors(model.gamma,correct))
    controls={}
    for label,source,correct in [('oracle',good,True),('shortcut',bad,False)]:
        result=scores(source.QubitModel().fit(records),records,correct);controls[label]=result
        assert result['parameter_relative_error']<.03 and result['calibration_chi2']<1.5,result
        assert (max(result['hidden'].values())<meta['prediction_limit'] if correct else min(result['hidden'].values())>meta['prediction_limit']),result
    rng=np.random.default_rng(meta['noise_seed']);parameters=[];chi2=[];oracle=[];shortcut=[]
    for _ in range(args.noise_trials):
        sample=records_at(exact+rng.normal(0,sigma,len(inputs)))
        a=good.QubitModel().fit(sample);b=bad.QubitModel().fit(sample)
        assert abs(a.gamma-b.gamma)<1e-12
        r=scores(a,sample,True);parameters.append(a.gamma);chi2.append(r['calibration_chi2'])
        oracle.extend(r['hidden'].values());shortcut.extend(errors(b.gamma,False).values())
    assert max(abs(np.array(parameters)/true-1))<.03 and np.mean(np.array(chi2)<1.5)>=.99
    assert max(oracle)<meta['prediction_limit'] and min(shortcut)>meta['prediction_limit']
    checks=[];refinement=[];trace=[];hermiticity=[];minimum=[];population=[];calibration=[]
    for gamma in [.04,.12,.25]:
        for i in range(8):
            e=ref.experiment(float(rng.uniform(0,10)),rng.uniform(-1.2,1.2,2),float(rng.uniform(0,.8)),rng.uniform(-np.pi,np.pi,(2,2)).tolist(),rng.uniform(-np.pi,np.pi,(2,2)).tolist())
            if i==0:e=ref.experiment(10,[1.2,1.2],.8)
            a=ref.predict([e],gamma,40)[0];b=ref.predict([e],gamma,56)[0]
            checks.append(float(abs(a-good.signal(good.signal_terms([e]),gamma)[0])))
            refinement.append(float(abs(a-b)))
            rho=ref.reduced_density(e,gamma,40)
            initial=np.kron(*(ref.spin_density(r) for r in e['preparation']))
            trace.append(float(abs(np.trace(rho)-1)));hermiticity.append(float(np.max(abs(rho-rho.conj().T))))
            minimum.append(float(np.linalg.eigvalsh(rho).min()));population.append(float(np.max(abs(np.diag(rho)-np.diag(initial)))))
        calibration.append(float(np.max(abs(good.signal(good.signal_terms(inputs),gamma)-bad.signal(bad.signal_terms(inputs),gamma)))))
    assert max(checks)<1e-9 and max(refinement)<1e-9,(max(checks),max(refinement))
    assert max(trace)<1e-12 and max(hermiticity)<1e-12 and min(minimum)>-1e-12 and max(population)<1e-12
    assert max(calibration)<1e-14
    report=dict(revision='neutral-r2',controls=controls,noise=dict(trials=args.noise_trials,parameter_range=[min(parameters),max(parameters)],relative_parameter_error_max=float(max(abs(np.array(parameters)/true-1))),calibration_chi2_max=max(chi2),calibration_pass_fraction=float(np.mean(np.array(chi2)<1.5)),oracle_hidden_error_max=max(oracle),shortcut_hidden_error_min=min(shortcut)),physical_checks=dict(independent_fock_oracle_error_max=max(checks),fock_40_to_56_difference_max=max(refinement),trace_error_max=max(trace),hermiticity_error_max=max(hermiticity),density_eigenvalue_min=min(minimum),population_change_max=max(population),calibration_closure_difference_max=max(calibration)),measurement_sigma_independent_of_parameter=sigma,prediction_limit=meta['prediction_limit'],seconds=time.time()-started)
    files=[TASK/'instruction.md',TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',ROOT/'scripts/qubit_control_baseline.py']
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
