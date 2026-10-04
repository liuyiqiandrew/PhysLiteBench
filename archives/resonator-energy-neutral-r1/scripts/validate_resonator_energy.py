"""Validate resonant slab energy against coupled mechanical finite elements."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from scipy.integrate import quad

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/resonator-energy'


def module(path):
    spec=importlib.util.spec_from_file_location('validation_'+str(abs(hash(str(path)))),path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def physical_checks(good,bad,ref):
    from scipy.integrate import quad
    hidden=sum(ref.hidden_inputs().values(),[]);cal=ref.calibration_inputs()
    truth=ref.predict(hidden);agreement=float(max(abs(good.predict_at(hidden,1.13)/truth-1)))
    refinement=float(max(abs(ref.predict(hidden,cells=2048)/truth-1)))
    cal_equivalence=float(max(abs(good.predict_at(cal,1.13)-bad.predict_at(cal,1.13))))
    matrix_error=[];flux_error=[];energy_min=[];field_error=[];corner_error=[]
    for stiffness in [.7,1.13,1.6]:
        for length in [.5,2.]:
            for mass in [0.,.2,1.2]:
                for ratio,resonance in [(.2,1.5),(.85,2.3)]:
                    e=ref.experiment(length,ratio*resonance,mass,resonance,.03)
                    energy,u,v=ref.solve(e,stiffness,cells=1024);density,U,V=good.wave_data(e,stiffness);omega=e['frequency']
                    corner_error.append(abs(good.predict_at([e],stiffness)[0]/energy-1));energy_min.append(energy)
                    reflection=u[0]/.03-1;transmission=u[-1]/.03
                    flux_error.append(abs(abs(reflection)**2+abs(transmission)**2-1))
                    if mass:field_error.append(float(max(abs(v-resonance**2/(resonance**2-omega**2)*u))))
                    # Independent quadrature of analytic standing-wave fields checks the source's closed spatial integrals.
                    q=omega*np.sqrt(density/stiffness);c,s=np.cos(q*length),np.sin(q*length);z=1j*omega*np.sqrt(2.)
                    aa,bb=np.linalg.solve([[z,stiffness*q],[-stiffness*q*s-z*c,stiffness*q*c-z*s]],[2*z*.03,0.])
                    uq=quad(lambda x:abs(aa*np.cos(q*x)+bb*np.sin(q*x))**2,0,length,epsabs=1e-13)[0]
                    vq=quad(lambda x:abs(q*(-aa*np.sin(q*x)+bb*np.cos(q*x)))**2,0,length,epsabs=1e-13)[0]
                    matrix_error.extend([abs(U-uq),abs(V-vq)])
    # Exact homogeneous matching and an impedance-matched resonant medium provide energy-velocity checks.
    matched=[];group_delay=[];scaling=[];low_frequency=[]
    for omega in [.3,.9,1.7]:
        e=ref.experiment(1.4,omega,amplitude=.03)
        matched.append(abs(good.predict_at([e],2.)[0]-.5*omega**2*.03**2*1.4))
    omega=1.2;resonance=2.;stiffness=1.13;mass=(2/stiffness-1)*(1-(omega/resonance)**2)
    e=ref.experiment(1.4,omega,mass,resonance,.03);density=2/stiffness
    derivative=2*mass*resonance**2*omega/(resonance**2-omega**2)**2
    qprime=np.sqrt(density/stiffness)+omega*derivative/(2*np.sqrt(stiffness*density))
    expected=.5*omega**2*np.sqrt(2.)*.03**2*1.4*qprime
    group_delay.append(abs(good.predict_at([e],stiffness)[0]/expected-1))
    for e in hidden[::3]:
        doubled=dict(e,incident_amplitude=2*e['incident_amplitude'])
        for source in [good,bad]:scaling.append(abs(source.predict_at([doubled],1.13)[0]/source.predict_at([e],1.13)[0]-4))
        small=dict(e,frequency=.0001)
        low_frequency.append(abs(good.predict_at([small],1.13)[0]/bad.predict_at([small],1.13)[0]-1))
    fits=[]
    for true in [.701,1.13,1.599]:
        exact=good.predict_at(cal,true);records=[dict(input=e,value=float(y),sigma=2e-7) for e,y in zip(cal,exact)]
        fits.append(abs(good.Model().fit(records).host_stiffness/true-1))
        profile=[float(np.sum((good.predict_at(cal,k)-exact)**2)) for k in np.linspace(.7,1.6,181)]
        index=int(np.argmin(profile));assert all(np.diff(profile[:index+1])<0) and all(np.diff(profile[index:])>0)
    for e in cal+hidden:
        assert .5<=e['length']<=2 and .3<=e['frequency']<=2 and 0<=e['resonator_mass']<=1.2
        assert 1.5<=e['resonance_frequency']<=2.5 and .01<=e['incident_amplitude']<=.04
        assert e['resonator_mass']==0 or e['frequency']/e['resonance_frequency']<=.85
    assert max(agreement,refinement,max(corner_error))<.0002 and cal_equivalence==0.
    assert max(flux_error)<1e-9 and max(field_error)<1e-8 and max(matrix_error)<1e-12 and min(energy_min)>0, (max(flux_error),max(field_error),max(matrix_error),min(energy_min))
    assert max(matched)<1e-14 and max(group_delay)<1e-12 and max(scaling)<1e-12 and max(low_frequency)<1e-7
    assert max(fits)<1e-7
    return {'oracle_coupled_FE1024_max_relative_error':agreement,'FE1024_to2048_relative_refinement':refinement,
        'full_domain_corner_reference_relative_error':max(corner_error),'calibration_control_difference':cal_equivalence,
        'scattering_flux_balance_max':max(flux_error),'coupled_resonator_displacement_relation_error':max(field_error),
        'analytic_spatial_integral_quadrature_error':max(matrix_error),'minimum_corner_energy':min(energy_min),
        'homogeneous_matched_energy_error':max(matched),'resonant_matched_energy_flux_group_delay_error':max(group_delay),
        'quadratic_amplitude_scaling_error_both_controls':max(scaling),'quasistatic_closure_relative_difference':max(low_frequency),
        'noiseless_fit_relative_errors':fits,'calibration_objective_full_domain_unimodal':True,'public_input_domain_verified':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--generate',action='store_true');parser.add_argument('--noise-trials',type=int,default=256)
    args=parser.parse_args();start=time.time()
    good=module(TASK/'solution/model.py');bad=module(ROOT/'scripts/resonator_energy_baseline.py');ref=module(TASK/'tests/reference.py')
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
        model=source.Model().fit(records);result={'parameter':model.host_stiffness,'parameter_relative_error':abs(model.host_stiffness/true-1),
            'calibration_chi2':chi(model,records),'hidden':scores(model)}
        assert result['calibration_chi2']<1.5 and result['parameter_relative_error']<.03
        assert (max(result['hidden'].values())<limit if label=='oracle' else min(result['hidden'].values())>limit)
        report['controls'][label]=result
    rng=np.random.default_rng(metadata['noise_seed']);parameters=[];chi2s=[];noise_oracle=[];noise_shortcut=[]
    for _ in range(args.noise_trials):
        sample=[dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,exact+rng.normal(0,sigma,len(inputs)))]
        a=good.Model().fit(sample);b=bad.Model().fit(sample);assert abs(a.host_stiffness-b.host_stiffness)<1e-10
        parameters.append(a.host_stiffness);chi2s.append(chi(a,sample))
        noise_oracle.append(max(scores(a).values()));noise_shortcut.append(min(scores(b).values()))
        assert noise_oracle[-1]<limit and noise_shortcut[-1]>limit
    assert max(chi2s)<1.5 and max(abs(np.array(parameters)/true-1))<.03
    report['noise']={'parameter_min':min(parameters),'parameter_max':max(parameters),'parameter_relative_error_max':float(max(abs(np.array(parameters)/true-1))),
        'calibration_chi2_max':max(chi2s),'calibration_pass_fraction':1.,'oracle_hidden_passes':args.noise_trials,'shortcut_hidden_passes':0,
        'oracle_hidden_max':max(noise_oracle),'shortcut_hidden_min':min(noise_shortcut)}
    extrema={}
    for label,source in [('oracle',good),('shortcut',bad)]:
        errors=[]
        for parameter in [min(parameters),max(parameters)]:
            model=source.Model();model.host_stiffness=parameter;errors.extend(scores(model).values())
        extrema[label]={'min':min(errors),'max':max(errors)}
    assert extrema['oracle']['max']<limit and extrema['shortcut']['min']>limit
    report['hidden_parameter_extrema_check']=extrema;report['physical_checks']=physical_checks(good,bad,ref);report['seconds']=time.time()-start
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TASK/'instruction.md',TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',TASK/'tests/metadata.json',TASK/'environment/data/calibration.json',ROOT/'scripts/resonator_energy_baseline.py',Path(__file__)]}
    output=ROOT/'jobs/resonator-energy-validation';output.mkdir(parents=True,exist_ok=True);(output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'results/resonator-energy-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('resonator-energy PASS',output/'summary.json',report['seconds'])
    print(json.dumps(report['controls'],indent=2));print(json.dumps(report['physical_checks'],indent=2))


if __name__=='__main__':main()
