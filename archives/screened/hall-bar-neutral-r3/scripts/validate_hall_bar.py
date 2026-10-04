"""Validate the closed moment-fluid Hall channel and local homogeneous readout."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from scipy.integrate import simpson

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/hall-bar'


def module(path):
    spec=importlib.util.spec_from_file_location('check_'+str(abs(hash(str(path)))),path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true')
    parser.add_argument('--noise-trials',type=int,default=256)
    args=parser.parse_args();start=time.time()
    good=module(TASK/'solution/model.py');bad=module(ROOT/'scripts/hall_bar_baseline.py');ref=module(TASK/'tests/reference.py')
    meta=json.loads((TASK/'tests/metadata.json').read_text());true=ref.TRUE_PARAMETER
    inputs=ref.calibration_inputs();exact=ref.predict(inputs,true);sigma=meta['measurement_sigma']
    def records_at(values):return [dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,values)]
    if args.generate:
        sample=records_at(exact+np.random.default_rng(meta['calibration_seed']).normal(0,sigma,len(inputs)))
        for directory in ['environment','tests']:
            (TASK/directory/'data/calibration.json').write_text(json.dumps(sample,indent=2)+'\n')
    records=json.loads((TASK/'environment/data/calibration.json').read_text())
    assert records==json.loads((TASK/'tests/data/calibration.json').read_text())
    hidden=ref.hidden_inputs();truth={n:ref.predict(es,true) for n,es in hidden.items()}
    def errors(model):
        return {n:float(np.linalg.norm(model.predict(es)-truth[n])/np.linalg.norm(truth[n])) for n,es in hidden.items()}
    def scores(model,sample):
        residual=(model.predict(inputs)-np.array([r['value'] for r in sample]))/sigma
        return dict(mobility=model.mobility,parameter_relative_error=abs(model.mobility/true-1),calibration_chi2=float(residual@residual)/(len(inputs)-1),hidden=errors(model))
    controls={}
    for label,source in [('oracle',good),('shortcut',bad)]:
        result=scores(source.Model().fit(records),records);controls[label]=result
        assert result['parameter_relative_error']<.03 and result['calibration_chi2']<1.5,result
        assert (max(result['hidden'].values())<.04 if label=='oracle' else max(result['hidden'].values())>.04),result
    rng=np.random.default_rng(meta['noise_seed']);parameters=[];chi=[];good_error=[];bad_error=[]
    group_errors={label:{name:[] for name in hidden} for label in ['oracle','shortcut']}
    calibration_equivalence=max(float(np.max(abs(np.array([good.response(e,mu) for e in inputs])-np.array([bad.response(e,mu) for e in inputs])))) for mu in [.05,.3,.65,1.,1.5])
    assert calibration_equivalence==0
    for _ in range(args.noise_trials):
        sample=records_at(exact+rng.normal(0,sigma,len(inputs)));a=good.Model().fit(sample);b=bad.Model().fit(sample)
        assert abs(a.mobility-b.mobility)<1e-12
        parameters.append(a.mobility)
        residual=(a.predict(inputs)-np.array([r['value'] for r in sample]))/sigma
        chi.append(float(residual@residual)/(len(inputs)-1))
        ga,gb=errors(a),errors(b);good_error.extend(ga.values());bad_error.append(max(gb.values()))
        for label,scored in [('oracle',ga),('shortcut',gb)]:
            for name,value in scored.items():group_errors[label][name].append(value)
    assert max(abs(np.array(parameters)/true-1))<.03
    assert np.mean(np.array(chi)<1.5)>=.99
    assert max(good_error)<.04 and min(bad_error)>.04
    comparisons=[];tensor=[];power=[];parity=[];refinement=[];closure_identity=[]
    for _ in range(16):
        e=ref.experiment(float(rng.uniform(.4,3)),.04,*rng.uniform(-3,3,3),'transverse_voltage',(-.7,.8))
        mu=float(rng.uniform(.05,1.5));sol=ref.solution(e,mu);y=np.linspace(-e['width']/2,e['width']/2,4001)
        z=sol.sol(y);b=ref.field(y,e);du=sol.sol(y,1)[0]
        pxy=z[1];pxx=2*b*ref.TAU2*pxy;pyy=-pxx
        # L+L^T, Lorentz rotation on both tensor indices, and relaxation.
        tensor.append(float(max(np.max(abs(2*b*pxy-pxx/ref.TAU2)),np.max(abs(-du+b*(pyy-pxx)-pxy/ref.TAU2)))))
        input_power=simpson(e['electric_field']*z[0],x=y)
        dissipated=simpson(z[0]**2/mu-pxy*du,x=y)
        power.append(float(abs(input_power-dissipated)/max(abs(input_power),1e-12)))
        target=ref.predict([e],mu)[0];actual=good.response(e,mu)
        comparisons.append(float(abs(target-actual)/max(abs(target),.001)))
        a,b=np.array(e['contacts'])*e['width']/2
        yc=np.linspace(a,b,2001);zc=sol.sol(yc)
        db=2*e['field_gradient']/e['width']-2*np.pi/e['width']*e['field_modulation']*np.sin(2*np.pi*yc/e['width'])
        omitted=simpson(-2*ref.TAU2*db*zc[1],x=yc)
        closure_identity.append(float(abs(target-bad.response(e,mu)-omitted)))
        fine=ref.solution(e,mu,1e-11);refinement.append(float(np.max(abs(sol.sol(y)-fine.sol(y)))))
        reverse={**e,'magnetic_field':-e['magnetic_field'],'field_gradient':-e['field_gradient'],'field_modulation':-e['field_modulation']}
        parity.append(float(abs(good.response(e,mu)+good.response(reverse,mu))))
    # The tensor residual includes the derivative of a tolerance-1e-9 BVP interpolant.
    assert max(comparisons)<.001 and max(tensor)<1e-8 and max(power)<1e-6 and max(refinement)<1e-8 and max(parity)<1e-12, (max(comparisons),max(tensor),max(power),max(refinement),max(parity))
    assert max(closure_identity)<1e-6,max(closure_identity)
    report=dict(revision='neutral-r3',controls=controls,noise=dict(trials=args.noise_trials,parameter_range=[min(parameters),max(parameters)],relative_parameter_error_max=float(max(abs(np.array(parameters)/true-1))),calibration_pass_fraction=float(np.mean(np.array(chi)<1.5)),calibration_chi2_max=max(chi),oracle_hidden_error_max=max(good_error),shortcut_worst_group_error_min=min(bad_error),group_error_ranges={label:{name:[min(values),max(values)] for name,values in groups.items()} for label,groups in group_errors.items()}),physical_checks=dict(independent_bvp_relative_error_max=max(comparisons),tensor_balance_error_max=max(tensor),power_balance_relative_error_max=max(power),bvp_refinement_difference_max=max(refinement),field_reversal_error_max=max(parity),local_homogeneous_missing_term_identity_error_max=max(closure_identity),calibration_equivalence_max=calibration_equivalence),measurement_sigma=sigma,calibration_records=len(records),shortcut_expected_failures=['gradient_field','modulated_field'],prediction_limit=.04,seconds=time.time()-start)
    files=[TASK/'instruction.md',TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',ROOT/'scripts/hall_bar_baseline.py']
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (ROOT/'results/hall-bar-r3-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
