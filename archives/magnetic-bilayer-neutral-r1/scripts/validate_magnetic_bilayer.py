"""Validate projected bilayer magnetostatics and the separated-source energy closure."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from scipy.integrate import quad

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/magnetic-bilayer'


def module(path):
    spec=importlib.util.spec_from_file_location('validation_'+str(abs(hash(str(path)))),path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def physical_checks(good,bad,ref):
    experiments=sum(ref.hidden_inputs().values(),[])
    agreement=max(abs(good.predict_at(experiments,.93)-ref.predict(experiments,.93)))
    poisson_error=[];symmetry=[];hermitian=[];energy_min=[];magnetic_min=[];decay_max=[];work_error=[];basis_residual=[]
    permutation=np.eye(4)[[2,3,0,1]]
    from scipy.linalg import expm
    rng=np.random.default_rng(19831)
    for k in [.3,.6,1.,1.5,2.]:
        for angle in np.linspace(-np.pi,np.pi,13):
            kx,ky=k*np.cos(angle),k*np.sin(angle)
            for gap in [.1,.4,1.]:
                energy=good.energy_matrix(kx,ky,gap)
                local=np.diag(np.repeat([.45,.70],2)+.08*k*k)
                magnetic=energy-local
                poisson_error.append(float(np.max(abs(magnetic+ref.field_matrix(kx,ky,gap)))))
                symmetry.append(float(np.max(abs(good.energy_matrix(-kx,-ky,gap)-energy.conj()))))
                symmetry.append(float(np.max(abs(permutation@magnetic@permutation-magnetic.conj()))))
                for source in [good,bad]:
                    a=source.energy_matrix(kx,ky,gap)
                    hermitian.append(float(np.max(abs(a-a.conj().T))))
                    energy_min.append(float(np.linalg.eigvalsh(a)[0]))
                    magnetic_min.append(float(np.linalg.eigvalsh(a-local)[0]))
                    rates,vectors,inverse=source.mode_basis(kx,ky,gap)
                    generator=(source.ROTATION-source.DAMPING*np.eye(4))@a/(1+source.DAMPING**2)
                    decay_max.append(float(max(rates.real)))
                    basis_residual.append(float(np.max(abs(generator@vectors-vectors*rates))))
                    state=rng.normal(size=4)+1j*rng.normal(size=4)
                    rate=np.vdot(state,a@generator@state).real
                    expected=-source.DAMPING/(1+source.DAMPING**2)*np.vdot(a@state,a@state).real
                    work_error.append(abs(rate-expected))
    # Direct Coulomb quadrature uses both volume charge and the boundary sheets.
    def coulomb(kx,ky,gap):
        k=np.hypot(kx,ky);intervals=[(0.,1.),(1+gap,2+gap)];matrix=np.zeros((4,4),complex)
        for a in range(4):
            ai,aj=intervals[a//2]
            for b in range(4):
                bi,bj=intervals[b//2]
                kernel=lambda z,w:np.exp(-k*abs(z-w))/(2*k)
                if a%2==0 and b%2==0:
                    def inner(z):
                        return quad(lambda w:kernel(z,w),bi,bj,points=[z] if bi<z<bj else None,epsabs=1e-11)[0]
                    value=ky*ky*quad(inner,ai,aj,epsabs=1e-11)[0]
                elif a%2==1 and b%2==1:
                    value=kernel(ai,bi)-kernel(ai,bj)-kernel(aj,bi)+kernel(aj,bj)
                elif a%2==0:
                    value=1j*ky*quad(lambda z:-kernel(z,bi)+kernel(z,bj),ai,aj,epsabs=1e-11)[0]
                else:
                    value=-1j*ky*quad(lambda w:-kernel(ai,w)+kernel(aj,w),bi,bj,epsabs=1e-11)[0]
                matrix[a,b]=value
        return matrix
    coulomb_errors=[]
    for kx,ky,gap in [(.31,.83,.27),(.57,-1.23,.61),(1.37,0.,.42)]:
        local=np.diag(np.repeat([.45,.70],2)+.08*(kx*kx+ky*ky))
        coulomb_errors.append(float(np.max(abs(coulomb(kx,ky,gap)-(good.energy_matrix(kx,ky,gap)-local)))))
    cal=ref.calibration_inputs();cal_equivalence=max(abs(good.predict_at(cal,.93)-bad.predict_at(cal,.93)))
    noiseless_fits=[];profile_minima=[]
    sigma=.00003
    for true in [.701,.93,1.299]:
        exact=good.predict_at(cal,true)
        records=[dict(input=e,value=float(y),sigma=sigma) for e,y in zip(cal,exact)]
        fitted=good.Model().fit(records).gyro_rate
        noiseless_fits.append(abs(fitted/true-1))
        profile=[float(np.sum((good.predict_at(cal,g)-exact)**2)) for g in np.linspace(.7,1.3,181)]
        # The objective is strictly decreasing then increasing over the full allowed interval.
        index=int(np.argmin(profile));profile_minima.append(index)
        assert all(np.diff(profile[:index+1])<0) and all(np.diff(profile[index:])>0)
    # Isolated-film and zero-time limits plus linear superposition.
    a=good.energy_matrix(.4,.9,60.);p=-np.expm1(-np.hypot(.4,.9))/np.hypot(.4,.9)
    separated=float(np.max(abs(a[:2,2:])))
    zero=max(abs(good.predict_at([dict(e,time=0.) for e in experiments],.93)))
    e=experiments[-1];e1=dict(e,initial_imag=[0.,0.]);e2=dict(e,initial_real=[0.,0.])
    linear=abs(good.predict_at([e],.93)[0]-sum(good.predict_at([e1,e2],.93)))
    assert agreement<1e-12 and max(poisson_error)<1e-12 and max(coulomb_errors)<1e-10
    assert max(symmetry)<1e-12 and max(hermitian)<1e-12 and min(energy_min)>.4 and min(magnetic_min)>-1e-12
    assert max(decay_max)<0 and max(work_error)<1e-11 and max(basis_residual)<1e-12
    assert cal_equivalence==0. and max(noiseless_fits)<1e-7 and separated<1e-20 and zero<1e-15 and linear<1e-15
    return {'oracle_poisson_dynamic_reference_max':float(agreement),'magnetostatic_poisson_matrix_max':max(poisson_error),
            'direct_coulomb_charge_integral_matrix_max':max(coulomb_errors),'signed_wavevector_and_layer_reflection_error':max(symmetry),
            'hermitian_reciprocity_error_both_controls':max(hermitian),'minimum_energy_eigenvalue_both_controls':min(energy_min),
            'minimum_magnetostatic_energy_eigenvalue_both_controls':min(magnetic_min),'maximum_dynamical_real_eigenvalue_both_controls':max(decay_max),
            'gilbert_energy_dissipation_residual_both_controls':max(work_error),'eigenbasis_residual_both_controls':max(basis_residual),
            'calibration_control_difference':float(cal_equivalence),'noiseless_fit_relative_errors':noiseless_fits,
            'calibration_objective_full_domain_unimodal':True,'large_separation_interlayer_coupling':separated,
            'zero_time_out_of_plane_signal_max':float(zero),'linear_superposition_error':float(linear)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--generate',action='store_true');parser.add_argument('--noise-trials',type=int,default=256)
    args=parser.parse_args();start=time.time()
    good=module(TASK/'solution/model.py');bad=module(ROOT/'scripts/magnetic_bilayer_baseline.py');ref=module(TASK/'tests/reference.py')
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
        model=source.Model().fit(records);result={'parameter':model.gyro_rate,'parameter_relative_error':abs(model.gyro_rate/true-1),
            'calibration_chi2':chi(model,records),'hidden':scores(model)}
        assert result['calibration_chi2']<1.5 and result['parameter_relative_error']<.03
        assert (max(result['hidden'].values())<limit if label=='oracle' else min(result['hidden'].values())>limit)
        report['controls'][label]=result
    rng=np.random.default_rng(metadata['noise_seed']);parameters=[];chi2s=[];noise_oracle=[];noise_shortcut=[]
    for _ in range(args.noise_trials):
        sample=[dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,exact+rng.normal(0,sigma,len(inputs)))]
        a=good.Model().fit(sample);b=bad.Model().fit(sample);assert abs(a.gyro_rate-b.gyro_rate)<1e-10
        parameters.append(a.gyro_rate);chi2s.append(chi(a,sample))
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
            model=source.Model();model.gyro_rate=parameter;errors.extend(scores(model).values())
        extrema[label]={'min':min(errors),'max':max(errors)}
    assert extrema['oracle']['max']<limit and extrema['shortcut']['min']>limit
    report['hidden_parameter_extrema_check']=extrema;report['physical_checks']=physical_checks(good,bad,ref);report['seconds']=time.time()-start
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TASK/'instruction.md',TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',TASK/'tests/metadata.json',TASK/'environment/data/calibration.json',ROOT/'scripts/magnetic_bilayer_baseline.py',Path(__file__)]}
    output=ROOT/'jobs/magnetic-bilayer-validation';output.mkdir(parents=True,exist_ok=True);(output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'results/magnetic-bilayer-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('magnetic-bilayer PASS',output/'summary.json',report['seconds'])
    print(json.dumps(report['controls'],indent=2));print(json.dumps(report['physical_checks'],indent=2))


if __name__=='__main__':main()
