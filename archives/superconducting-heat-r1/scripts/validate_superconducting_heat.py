"""Validate phase-biased BCS heat tunneling and the gap-edge coherence control."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from scipy.integrate import quad
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/superconducting-heat'


def module(path):
    spec=importlib.util.spec_from_file_location('validation_'+str(abs(hash(str(path)))),path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def physical_checks(good,bad,ref):
    experiments=sum(ref.hidden_inputs().values(),[])
    agreement=max(abs(good.predict_at(experiments,45.)-ref.predict(experiments,45.)))
    refined=max(abs(ref.unit_power(e['left_temperature'],e['right_temperature'],e['phase'],1e-13)-ref.unit_power(e['left_temperature'],e['right_temperature'],e['phase']))*45 for e in experiments)
    from numpy.polynomial.legendre import leggauss
    original=good.predict_at(experiments,45.)
    good.NODES,good.WEIGHTS=leggauss(240);good.power_per_conductance.cache_clear()
    refinement=max(abs(good.predict_at(experiments,45.)-original))
    good.NODES,good.WEIGHTS=leggauss(160);good.power_per_conductance.cache_clear()
    gap_differences=[];positive=[];phase_symmetry=[];coherence=[];branch_positive=[]
    for tl in np.linspace(.25,1.05,9):
        for tr in np.linspace(.25,1.05,9):
            dl,dr=ref.material_gaps(tl,tr);gap_differences.append(dr-dl)
            for phase in [-np.pi,-.9,0.,.9,np.pi]:
                e=ref.experiment(tl,tr,phase)
                for source in [good,bad]:
                    q=source.predict_at([e],45.)[0]
                    positive.append(q*(1/tr-1/tl))
                    phase_symmetry.append(abs(q-source.predict_at([dict(e,phase=-phase)],45.)[0]))
                for energy in [max(dl,dr)*(1+1e-6),max(dl,dr)*1.5,max(dl,dr)*4]:
                    w=ref.branch_weight(energy,dl,dr,phase)
                    coherence.append(abs(w-(1-dl*dr/energy**2*np.cos(phase))))
                    branch_positive.append(w)
    corners=[ref.experiment(a,b,p) for a,b in [(.25,1.05),(1.05,.25),(.25,.25),(1.05,1.05)] for p in [0.,np.pi,.7]]
    corner_error=max(abs(good.predict_at(corners,45.)-ref.predict(corners,45.)))
    cal=ref.calibration_inputs();cal_equivalence=max(abs(good.predict_at(cal,45.)-bad.predict_at(cal,45.)))
    # Independent normal-state branch sum and Fermi integral recover the Wiedemann-Franz heat law.
    normal_checks=[]
    for tl,tr in [(.4,.8),(.95,.3)]:
        integral=quad(lambda energy:energy*ref.branch_weight(energy,0.,0.,.7)*(expit(-energy/tl)-expit(-energy/tr)),0.,np.inf,epsabs=1e-12)[0]
        expected=np.pi**2/12*(tl*tl-tr*tr)
        normal_checks.append(abs(integral-expected))
    # Boltzmann edge concentration gives the source ratio in the low-temperature limit.
    low_t=[]
    for temp in [.04,.01]:
        dl,dr=ref.material_gaps(temp,temp/2);hi=max(dl,dr);lo=min(dl,dr)
        def weight(s):
            en=np.sqrt(hi*hi+s*s)
            return np.exp(-(en-hi)/temp)/np.sqrt(en*en-lo*lo)
        qp=quad(lambda s:(hi*hi+s*s)*weight(s),0.,5*np.sqrt(hi*temp),epsabs=1e-12)[0]
        cross=quad(lambda s:dl*dr*weight(s),0.,5*np.sqrt(hi*temp),epsabs=1e-12)[0]
        low_t.append(abs(cross/qp-dl*dr/hi**2))
    assert agreement<1e-9 and refined<1e-9 and refinement<1e-9 and corner_error<1e-9
    assert min(gap_differences)>.9 and min(positive)>-1e-14 and max(phase_symmetry)<1e-14
    assert max(coherence)<1e-12 and min(branch_positive)>0 and cal_equivalence<1e-13
    assert max(normal_checks)<1e-12 and low_t[1]<low_t[0] and low_t[1]<.005
    return {'oracle_branch_amplitude_reference_max_pW':float(agreement),'reference_tolerance_refinement_pW':float(refined),
            'oracle160_vs240_nodes_pW':float(refinement),'domain_corner_reference_error_pW':float(corner_error),
            'minimum_gap_separation_K':min(gap_differences),'minimum_entropy_production_pW_per_K_both_controls':min(positive),
            'phase_reversal_error_pW_both_controls':max(phase_symmetry),'branch_sum_vs_reduced_coherence_max':max(coherence),
            'minimum_branch_probability':min(branch_positive),'calibration_control_difference_pW':float(cal_equivalence),
            'normal_state_fermi_integral_error_K2':max(normal_checks),'low_temperature_coherence_ratio_errors':low_t,
            'fixed_phase_voltage_work_W':0.,'order_note':'All predictions are the specified leading tunneling order; separated unequal gaps avoid coincident-edge perturbative singularities.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--generate',action='store_true');parser.add_argument('--noise-trials',type=int,default=256)
    args=parser.parse_args();start=time.time()
    good=module(TASK/'solution/model.py');bad=module(ROOT/'scripts/superconducting_heat_baseline.py');ref=module(TASK/'tests/reference.py')
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
    def scores(model):return {name:float(np.sqrt(np.mean((model.predict(es)-truth[name])**2)/np.mean(truth[name]**2))) for name,es in hidden.items()}
    report={'revision':1,'calibration_seed':metadata['calibration_seed'],'noise_seed':metadata['noise_seed'],'noise_trials':args.noise_trials,'controls':{}}
    for label,source in [('oracle',good),('shortcut',bad)]:
        model=source.Model().fit(records);result={'parameter':model.conductance,'parameter_relative_error':abs(model.conductance/true-1),
            'calibration_chi2':chi(model,records),'hidden':scores(model)}
        assert result['calibration_chi2']<1.5 and result['parameter_relative_error']<.03
        assert (max(result['hidden'].values())<limit if label=='oracle' else min(result['hidden'].values())>limit)
        report['controls'][label]=result
    rng=np.random.default_rng(metadata['noise_seed']);parameters=[];chi2s=[]
    for _ in range(args.noise_trials):
        sample=[dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,exact+rng.normal(0,sigma,len(inputs)))]
        a=good.Model().fit(sample);b=bad.Model().fit(sample);assert abs(a.conductance-b.conductance)<1e-10
        parameters.append(a.conductance);chi2s.append(chi(a,sample))
    assert max(chi2s)<1.5 and max(abs(np.array(parameters)/true-1))<.03
    report['noise']={'parameter_min':min(parameters),'parameter_max':max(parameters),'parameter_relative_error_max':float(max(abs(np.array(parameters)/true-1))),
        'calibration_chi2_max':max(chi2s),'calibration_pass_fraction':1.}
    extrema={}
    for label,source in [('oracle',good),('shortcut',bad)]:
        errors=[]
        for parameter in [min(parameters),max(parameters)]:
            model=source.Model();model.conductance=parameter;errors.extend(scores(model).values())
        extrema[label]={'min':min(errors),'max':max(errors)}
    assert extrema['oracle']['max']<limit and extrema['shortcut']['min']>limit
    report['hidden_parameter_extrema_check']=extrema;report['physical_checks']=physical_checks(good,bad,ref);report['seconds']=time.time()-start
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TASK/'instruction.md',TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',TASK/'tests/metadata.json',TASK/'environment/data/calibration.json',ROOT/'scripts/superconducting_heat_baseline.py',Path(__file__)]}
    output=ROOT/'jobs/superconducting-heat-validation';output.mkdir(parents=True,exist_ok=True);(output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'results/superconducting-heat-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('superconducting-heat PASS',output/'summary.json',report['seconds'])
    print(json.dumps(report['controls'],indent=2));print(json.dumps(report['physical_checks'],indent=2))


if __name__=='__main__':main()
