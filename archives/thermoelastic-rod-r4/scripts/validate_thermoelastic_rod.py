"""Validate the r4 caloric closure and controls; regenerate data only explicitly."""
import argparse
import hashlib
import importlib.util
import itertools
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/thermoelastic-rod'


def module(path):
    spec = importlib.util.spec_from_file_location('validation_'+hashlib.sha256(str(path).encode()).hexdigest()[:12], path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def metrics(model, records, reference, hidden, truth):
    residual = (model.predict([r['input'] for r in records])-np.array([r['value'] for r in records]))/np.array([r['sigma'] for r in records])
    return dict(parameter=model.conductivity, parameter_relative_error=abs(model.conductivity/reference.TRUE_PARAMETER-1),
                calibration_chi2=float(residual@residual)/(len(records)-1),
                hidden={name:float(np.sqrt(np.mean((model.predict(inputs)-truth[name])**2))) for name,inputs in hidden.items()})


def physical_checks(mod, chi, contact):
    n = mod.CELLS
    g,x,b0,b1,st,sz,heat = mod.operators(145.,contact,chi)
    identity = np.eye(2*n+1)
    theta, z, body = identity[:n], mod.STRAIN_SCALE*identity[n:2*n], identity[-1]
    strain = st@theta+sz@z
    stress = (mod.MODULUS_0+mod.MODULUS_1)*strain-mod.MODULUS_1*z-(b0+b1)[:,None]*theta
    volume = mod.AREA*mod.LENGTH/n
    physical_entropy_T0 = mod.HEAT_CAPACITY*theta+mod.TEMPERATURE*((b0+b1)[:,None]*strain-b1[:,None]*z)
    relaxed_entropy_T0 = (mod.HEAT_CAPACITY+mod.TEMPERATURE*b1**2/mod.MODULUS_1)[:,None]*theta+mod.TEMPERATURE*b0[:,None]*strain
    w = volume*np.sum(physical_entropy_T0,axis=0)+mod.BODY_CAPACITY*body
    wr = volume*np.sum(relaxed_entropy_T0,axis=0)+mod.BODY_CAPACITY*body
    q = volume*(mod.MODULUS_0*strain.T@strain+mod.MODULUS_1*(strain-z).T@(strain-z)+mod.HEAT_CAPACITY/mod.TEMPERATURE*theta.T@theta)+mod.BODY_CAPACITY/mod.TEMPERATURE*np.outer(body,body)
    eta=mod.material(x,chi)[2]
    temperature=np.vstack([theta,body])
    zrate=z@g
    dissipation=temperature.T@heat@temperature/mod.TEMPERATURE+volume*zrate.T@(eta[:,None]*zrate)
    residual=g.T@q+q@g+2*dissipation
    initial=mod.preparation(chi)
    return dict(chi=chi,contact=contact,
                maximum_growth_rate=float(np.max(np.linalg.eigvals(g).real)),
                maximum_mean_strain=float(np.max(np.abs(np.mean(strain,axis=0)))),
                stress_relative_nonuniformity=float(np.max(np.ptp(stress,axis=0))/mod.MODULUS_0),
                physical_energy_generator_residual=float(np.max(np.abs(w@g))),
                relaxed_energy_generator_residual=float(np.max(np.abs(wr@g))),
                physical_availability_min_eigenvalue=float(np.linalg.eigvalsh(q)[0]),
                dissipation_identity_relative_residual=float(np.linalg.norm(residual)/max(np.linalg.norm(dissipation),1e-20)),
                initial_z_rate=float(np.max(np.abs(zrate@initial))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regenerate',action='store_true')
    args=parser.parse_args()
    started=time.perf_counter()
    oracle=module(TASK/'solution/model.py')
    shortcut=module(ROOT/'scripts/thermoelastic_rod_baseline.py')
    ref=module(TASK/'tests/reference.py')
    meta=json.loads((TASK/'tests/metadata.json').read_text())
    inputs=ref.calibration_inputs()*meta['calibration_repetitions']
    sigma=meta['measurement_sigma']
    noiseless=ref.predict(inputs,ref.TRUE_PARAMETER,cells=320)
    if args.regenerate:
        values=noiseless+np.random.default_rng(meta['calibration_seed']).normal(0,sigma,len(inputs))
        records=[dict(input=e,value=float(y),sigma=sigma) for e,y in zip(inputs,values)]
        data=json.dumps(records,indent=2)+'\n'
        for area in ['environment','tests']:(TASK/area/'data/calibration.json').write_text(data)
    public=TASK/'environment/data/calibration.json'
    private=TASK/'tests/data/calibration.json'
    assert public.read_bytes()==private.read_bytes()
    records=json.loads(public.read_text())
    assert [r['input'] for r in records]==inputs
    assert all(set(r)=={'input','value','sigma'} and r['sigma']==sigma for r in records)
    hidden=ref.hidden_inputs()
    truth={name:ref.predict(es,ref.TRUE_PARAMETER) for name,es in hidden.items()}
    report=dict(status='validation_in_progress',revision=4,data_integrity=dict(records=len(inputs),
                distinct_settings=len(ref.calibration_inputs()),repetitions=meta['calibration_repetitions'],
                fixed_instrument_sigma=sigma,public_private_identical=True,
                noiseless_generation='Independent entropy-coordinate reference,320cells; finite-resolution error checked separately.'),controls={})
    for name,mod in [('oracle',oracle),('shortcut',shortcut)]:
        result=metrics(mod.Model().fit(records),records,ref,hidden,truth)
        report['controls'][name]=result
        assert result['parameter_relative_error']<.03 and result['calibration_chi2']<1.5
        assert (max(result['hidden'].values())<.025 if name=='oracle' else min(result['hidden'].values())>.025)
    cal_error=float(np.max(np.abs(oracle.Model(ref.TRUE_PARAMETER).predict(inputs)-noiseless)))
    report['data_integrity']['oracle_deterministic_calibration_max_error']=cal_error
    assert cal_error<sigma/2
    print('Nominal controls passed; calibration deterministic error',cal_error,flush=True)

    # Whole-domain structured cases and independent random interiors.
    setups=[(k,c,h) for k,c,h in itertools.product([80.,145.,220.],[0.,.01,.25,.6,1.],[0.,.4,1.2])]
    rng=np.random.default_rng(4094)
    setups += [(float(rng.uniform(80,220)),float(rng.uniform(0,1)),float(rng.uniform(0,1.2))) for _ in range(12)]
    preps=[(.2,0.,.8,-.6),(-.3,.3,.4,.8),(0.,-.8,0.,0.)]
    domain=[]
    for k,c,h in setups:
        es=[dict(zip(('mean','first','second','bath_initial'),p),chi=c,contact=h,time=t,observable=o)
            for p,t,o in itertools.product(preps,[0.,.15,1.,5.,20.,80.],['mean','first','second','bath'])]
        actual=oracle.Model(k).predict(es)
        reference=ref.predict(es,k)
        source=shortcut.Model(k).predict(es)
        agreement=float(np.max(np.abs(actual-reference)))
        assert agreement<.002 and np.isfinite(source).all()
        if c==0:assert np.max(np.abs(actual-source))<1e-9
        starts=[e for e in es if e['time']==0]
        expected=np.array([{'mean':e['mean'],'first':e['first']/2,'second':e['second']/2,'bath':e['bath_initial']}[e['observable']] for e in starts])
        zero_error=float(np.max(np.abs(oracle.Model(k).predict(starts)-expected)))
        source_growth=float(np.max(np.linalg.eigvals(shortcut.operators(k,h,c)[0]).real))
        assert zero_error<1e-9 and source_growth<1e-9
        domain.append(dict(conductivity=k,chi=c,contact=h,reference_max_error=agreement,zero_time_error=zero_error,source_maximum_growth_rate=source_growth,source_max_difference=float(np.max(np.abs(actual-source)))))
    report['domain']=dict(structured_systems=45,random_systems=12,readouts_per_system=72,cases=domain)
    print('Domain checks passed',len(domain),flush=True)

    refinements=[]
    for k in [80.,145.,220.]:
        for name,es in hidden.items():
            nominal=oracle.Model(k).predict(es)
            refined=oracle.predict_values(es,k,cells=160)
            r160=ref.predict(es,k,160)
            r320=ref.predict(es,k,320)
            row=dict(conductivity=k,group=name,oracle_80_to_160=float(np.max(np.abs(nominal-refined))),
                     reference_160_to_320=float(np.max(np.abs(r160-r320))),oracle_to_ref320=float(np.max(np.abs(nominal-r320))),
                     per_case_absolute_error=np.abs(nominal-r320).tolist())
            assert max(row['oracle_80_to_160'],row['oracle_to_ref320'])<.0006 and row['reference_160_to_320']<.0002
            refinements.append(row)
    report['per_graded_case_refinement']=refinements
    report['physical_checks']={name:[physical_checks(mod,c,h) for c,h in itertools.product([0.,.01,.5,1.],[0.,.4,1.2])] for name,mod in [('oracle',oracle),('shortcut',shortcut)]}
    for r in report['physical_checks']['oracle']:
        assert r['physical_energy_generator_residual']<1e-7 and r['dissipation_identity_relative_residual']<1e-10
        assert r['physical_availability_min_eigenvalue']>0
    for r in report['physical_checks']['shortcut']:
        assert r['relaxed_energy_generator_residual']<1e-7
    for rows in report['physical_checks'].values():
        for r in rows: assert r['maximum_growth_rate']<1e-9 and r['initial_z_rate']<1e-12 and r['stress_relative_nonuniformity']<1e-12 and r['maximum_mean_strain']<1e-12

    # Global sampled profiles; no claim about arbitrary noisy objectives.
    cal=ref.calibration_inputs()
    grid=np.linspace(80,220,81)
    predictions=np.array([oracle.Model(k).predict(cal) for k in grid])
    equality=max(float(np.max(np.abs(oracle.Model(k).predict(cal)-shortcut.Model(k).predict(cal)))) for k in grid)
    derivative=np.diff(predictions,axis=0)/np.diff(grid)[:,None]
    positive_amplitude=[j for j,e in enumerate(cal) if e['first']>0]
    assert np.max(derivative[:,positive_amplitude])<0 and equality<1e-9
    recovery=[]
    for k in np.linspace(80,220,17):
        values=oracle.Model(k).predict(cal)
        sample=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(cal,values)]
        model=oracle.Model().fit(sample)
        objective=np.sum((predictions-values)**2,axis=1)
        local_minima=int(np.sum((objective[1:-1]<objective[:-2])&(objective[1:-1]<objective[2:])))
        assert abs(model.conductivity/k-1)<1e-6 and local_minima<=1
        recovery.append(dict(true=float(k),fitted=model.conductivity,sampled_local_minima=local_minima))
    gap_rows=[]
    for k in np.linspace(80,220,41):
        for name,es in hidden.items():
            target=oracle.Model(k).predict(es)
            wrong=shortcut.Model(k).predict(es)
            gap_rows.append(dict(conductivity=float(k),group=name,rmse=float(np.sqrt(np.mean((target-wrong)**2))),rms_signal=float(np.sqrt(np.mean(target**2)))))
    assert min(r['rmse'] for r in gap_rows)>.028
    report['identifiability']=dict(grid=grid.tolist(),predictions=predictions.tolist(),recoveries=recovery,
                                  positive_amplitude_maximum_sampled_derivative=float(np.max(derivative[:,positive_amplitude])),
                                  exact_calibration_control_difference=equality,qualification='Finite global grid and endpoint/interior recovery; no analytic monotonicity claim.')
    report['full_parameter_group_gaps']=gap_rows
    print('Refinement, physics and identification passed',flush=True)

    rng=np.random.default_rng(meta['noise_seed'])
    noise=[]
    for index in range(256):
        values=noiseless+rng.normal(0,sigma,len(inputs))
        sample=[dict(input=e,value=float(v),sigma=sigma) for e,v in zip(inputs,values)]
        fitted=oracle.Model().fit(sample)
        row=metrics(fitted,sample,ref,hidden,truth)
        other=shortcut.Model(fitted.conductivity)
        row['shortcut_hidden']={name:float(np.sqrt(np.mean((other.predict(es)-truth[name])**2))) for name,es in hidden.items()}
        assert row['parameter_relative_error']<.03 and row['calibration_chi2']<1.5
        assert max(row['hidden'].values())<.025 and min(row['shortcut_hidden'].values())>.025
        if index in [0,255]:assert abs(shortcut.Model().fit(sample).conductivity-fitted.conductivity)<1e-6
        noise.append(row)
        if (index+1)%64==0:print('Noise draws',index+1,flush=True)
    report['noise']=dict(draws=256,seed=meta['noise_seed'],oracle_passes=256,shortcut_rejections=256,
                         oracle_hidden_max=max(max(r['hidden'].values()) for r in noise),
                         shortcut_hidden_min=min(min(r['shortcut_hidden'].values()) for r in noise),
                         calibration_chi2_max=max(r['calibration_chi2'] for r in noise),
                         parameter_relative_error_max=max(r['parameter_relative_error'] for r in noise),rows=noise)
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TASK/'environment/README.md',TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',TASK/'tests/test_hidden.py',TASK/'tests/metadata.json',ROOT/'scripts/thermoelastic_rod_baseline.py',Path(__file__).resolve()]}
    report['status']='passed'
    report['seconds']=time.perf_counter()-started
    output=ROOT/'results/thermoelastic-rod-r4-validation.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'passed','seconds':report['seconds'],'noise':{k:v for k,v in report['noise'].items() if k!='rows'}},indent=2))


if __name__=='__main__':main()
