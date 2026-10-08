"""Validate fast-contact entropy revision 7 and its paired finite-mass calorimetry."""
import argparse
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/entropy-anomaly'
ARCHIVE = ROOT/'archives/entropy-anomaly-r6'
RESULTS = ROOT/'results'
NOISE_SEED = 9161066
ANCHORS = ['bulk_anchor', 'uniform_port_anchor']


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def error(actual, expected):
    return float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_controls():
    output = {}
    for label, source in [('oracle',TASK/'solution/model.py'),('shortcut',ROOT/'scripts/entropy_anomaly_baseline.py')]:
        folder = RESULTS/'entropy-anomaly-r7-controls'/label
        app = folder/'app'; app.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,app/'model.py'); shutil.copyfile(TASK/'environment/test_public.py',app/'test_public.py')
        (app/'data').mkdir(exist_ok=True); shutil.copyfile(TASK/'environment/data/calibration.json',app/'data/calibration.json')
        env = dict(os.environ,PYTHONPATH=str(app),PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OPENBLAS_NUM_THREADS='1',METRICS_PATH=str(folder/'metrics.json'))
        start = time.monotonic()
        run = subprocess.run([sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',str(app/'test_public.py'),str(TASK/'tests/test_hidden.py')],cwd=app,env=env,text=True,capture_output=True,timeout=60)
        (folder/'pytest.txt').write_text(run.stdout+run.stderr)
        metrics = json.loads((folder/'metrics.json').read_text())
        assert metrics['calibration_chi2']<1.5 and metrics['parameter_relative_error']<.03
        for key,value in metrics['hidden'].items():
            assert value<.04 if key in ANCHORS or label=='oracle' else value>.04
        expected = '9 passed' if label=='oracle' else '3 failed, 6 passed'
        assert expected in run.stdout and run.returncode==(0 if label=='oracle' else 1),run.stdout+run.stderr
        output[label] = dict(exit_code=run.returncode,seconds=time.monotonic()-start,expected=expected,metrics=metrics)
    return output


def uniform_covariance_entropy(e,gamma,mass):
    g = np.diag([gamma,gamma*e['drag_ratio']])
    friction = np.array([[gamma,-e['magnetic']],[e['magnetic'],gamma*e['drag_ratio']]])
    mean = np.linalg.solve(friction,[e['force_x'],e['force_y']])
    relaxation = np.kron(friction,np.eye(2))+np.kron(np.eye(2),friction)
    switch = e['switch_rate']*np.eye(4)
    temperatures = e['temperature']*(1+e['bath_contrast']*np.array([1.,-1.]))
    system = np.block([[relaxation+switch,-switch],[-switch,relaxation+switch]])
    cov = np.linalg.solve(system,np.r_[(g*temperatures[0]).ravel(),(g*temperatures[1]).ravel()]).reshape(2,2,2)
    leak = sum(np.sum(np.diag(g)*(np.diag(c)/t-.5)) for c,t in zip(cov,temperatures))
    excess = mean@g@mean/(e['temperature']*(1-e['bath_contrast']**2))
    return float(leak/mass+excess),float(leak),float(excess)


def isotropic_gradient_formula(e,gamma):
    ratio = e['switch_rate']/gamma; delta = e['bath_contrast']
    source = 2*e['temperature']*e['wavenumber']**2*(1-np.sqrt(1-e['contrast']**2))/(3*gamma*(1-delta**2))
    factor = 1+delta**2*(2*ratio-1)*(ratio+3)/((3+2*ratio)*(1+ratio)**2)
    return source*factor,source,factor


def spatial_port_checks(ref,oracle,shortcut,previous):
    oldref=module(ARCHIVE/'tasks/entropy-anomaly/tests/reference.py','bulk_r6_reference')
    bulk=sum(oldref.hidden_inputs().values(),[])
    rng=np.random.default_rng(710607)
    for _ in range(64):
        bulk.append(ref.experiment(rng.uniform(-1.5,1.5),rng.uniform(-1.5,1.5),rng.uniform(.25,8),rng.uniform(-5,5),rng.uniform(.8,1.4),rng.uniform(0,.65),int(rng.integers(1,4)),rng.uniform(0,.65),rng.uniform(.2,2)))
    bulk_errors=[];complements=[];phase_periodicity=[];equal_contacts=[];uniform=[];lag_anchors=[]
    weighted_values=[];density_lag=[]
    for gamma in [.7,1.1,1.6]:
        expected=previous.coefficients(bulk,gamma)
        bulk_errors.extend(abs(oracle.coefficients(bulk,gamma)-expected));bulk_errors.extend(abs(shortcut.coefficients(bulk,gamma)-expected))
        for e in bulk:
            first=dict(e,detector_phase=.7);other=dict(e,detector_phase=.7-np.pi)
            for source in [oracle,shortcut]:
                values=source.coefficients([first,other],gamma)
                complements.append(abs(sum(values)-source.coefficients([e],gamma)[0]))
                phase_periodicity.append(abs(np.diff(source.coefficients([dict(e,detector_phase=-np.pi),dict(e,detector_phase=np.pi)],gamma))[0]))
                weighted_values.extend(values)
            zero=dict(first,bath_contrast=0.)
            equal_contacts.append(abs(oracle.coefficients([zero],gamma)[0]-shortcut.coefficients([zero],gamma)[0]))
            flat=dict(first,contrast=0.)
            uniform.extend([abs(oracle.coefficients([flat],gamma)[0]-.5*previous.coefficients([dict(e,contrast=0.)],gamma)[0]),abs(oracle.coefficients([flat],gamma)[0]-shortcut.coefficients([flat],gamma)[0])])
            if e['drag_ratio']==1. and e['magnetic']==0.:
                state=oracle.limiting_state(e,gamma)
                delta,kappa=e['bath_contrast'],e['switch_rate'];weight=(1+np.cos(state['phase']-.7))/2
                temperature=e['temperature']*(1+e['contrast']*np.cos(state['phase']))
                factor=delta**2*gamma**2/((gamma+kappa)**2*(gamma+2*kappa)*(1-delta**2))
                closed=factor*state['dx']*weight@(state['derivative']@state['derivative']@(temperature*state['density']))
                lag_anchors.append(abs(oracle.coefficients([first],gamma)[0]-shortcut.coefficients([first],gamma)[0]-closed))
                density_lag.append(float(np.max(abs(state['density_correction'][0,0]))))
    # Independent scalar no-magnetic/isotropic lag anchors with driven and
    # zero-drive profiles. The predicted correction vanishes for no x drive.
    for gamma,force,delta,rate,a,k in itertools.product([.7,1.1,1.6],[0.,.8],[0.,.65],[.2,2.],[.4,.65],[1,3]):
        e=ref.experiment(force,.2,1.,0.,1.,a,k,delta,rate);state=oracle.limiting_state(e,gamma)
        for phase in [-np.pi/2,0.,np.pi/2]:
            port=dict(e,detector_phase=phase)
            coefficient=delta**2*gamma**2/((gamma+rate)**2*(gamma+2*rate)*(1-delta**2))
            expected=coefficient*force*k/2*state['dx']*state['density']@np.sin(state['phase']-phase)
            lag_anchors.append(abs(oracle.coefficients([port],gamma)[0]-shortcut.coefficients([port],gamma)[0]-expected))
    # Match the counted port entropy against an independent weighted energy
    # transport identity, including the boundary transport term d_x(w/T).
    weighted_energy_errors=[];residence=[];background_scaling=[]
    cases=[ref.hidden_inputs()['isotropic_drive'][0],ref.hidden_inputs()['anisotropic_contacts'][0],ref.hidden_inputs()['magnetic_drive'][0]]
    for e in cases:
        gamma=1.1;mass=.002*(gamma*min(1.,e['drag_ratio']))**2/(e['temperature']*e['wavenumber']**2)
        phase,temperatures,states,basis=ref.kinetic_state(e,gamma,mass,degree=16,spatial=49)
        weight=(1+np.cos(phase-e['detector_phase']))/2;dw=-e['wavenumber']*np.sin(phase-e['detector_phase'])/2
        energies=[basis*(c[0,0]+np.sqrt(2)/2*(c[2,0]+c[0,2])) for c in states]
        transported=0.;direct=0.
        for sign,(temperature,c) in enumerate(zip(temperatures,states)):
            dt=-e['temperature']*(1+[1,-1][sign]*e['bath_contrast'])*e['contrast']*e['wavenumber']*np.sin(phase)
            jx,jy=np.sqrt(basis/mass)*c[1,0],np.sqrt(basis/mass)*c[0,1]
            flux=basis**1.5/(2*np.sqrt(mass))*(4*c[1,0]+np.sqrt(6)*c[3,0]+np.sqrt(2)*c[1,2])
            transported+=2*np.pi*np.mean(weight*(e['force_x']*jx+e['force_y']*jy)/temperature+flux*(dw/temperature-weight*dt/temperature**2)+e['switch_rate']/mass*weight*(energies[1-sign]-energies[sign])/temperature)
            for drag,pair in [(gamma,(2,0)),(gamma*e['drag_ratio'],(0,2))]:
                direct+=2*np.pi*drag/mass*np.mean(weight*(basis*(c[0,0]+np.sqrt(2)*c[pair])/temperature-c[0,0]))
        measured_residence=2*np.pi*np.mean(weight*(states[0][0,0]+states[1][0,0]))
        measured_background=ref.finite_mass_entropy(ref.background_experiment(e),gamma,mass,degree=16,spatial=49)
        paired=ref.finite_mass_excess(e,gamma,mass,degree=16,spatial=49)
        weighted_energy_errors.extend([abs(transported-direct),abs(paired-(direct-measured_background*measured_residence))])
        residence.append(measured_residence)
        other=ref.finite_mass_entropy(ref.background_experiment(e),gamma,mass/2,degree=16,spatial=49)
        background_scaling.append(abs(measured_background*mass-other*mass/2))
    errors=bulk_errors+complements+phase_periodicity+equal_contacts+uniform+lag_anchors+weighted_energy_errors+background_scaling
    assert max(errors)<2e-7 and all(0<r<1 for r in residence),max(errors)
    return dict(bulk_preservation_cases=len(bulk),bulk_preservation_max_abs=float(max(bulk_errors)),complementary_port_sum_error=float(max(complements)),
                detector_phase_periodicity_error=float(max(phase_periodicity)),equal_contact_port_source_error=float(max(equal_contacts)),
                uniform_profile_half_bulk_error=float(max(uniform)),isotropic_closed_label_lag_anchor_error=float(max(lag_anchors)),
                weighted_energy_transport_identity_error=float(max(weighted_energy_errors)),actual_finite_mass_weighted_residences=residence,
                uniform_background_inverse_mass_scaling_error=float(max(background_scaling)),minimum_port_excess_observed=float(min(weighted_values)),
                no_port_positivity_assumed=True,seed=710607)


def physics_checks(ref,oracle,shortcut,previous,truths):
    start = time.monotonic(); gamma=ref.TRUE_PARAMETER
    inputs = sum(ref.hidden_inputs().values(),[]); truth=np.concatenate(list(truths.values()))
    direct = oracle.coefficients(inputs,gamma)
    half_mass = ref.predict(inputs,gamma,epsilon=.003)
    print('entropy r7 half-mass reference complete',flush=True)
    refined = ref.predict(inputs,gamma,degree=16,spatial=49)
    checks = dict(oracle_reference_max_abs=float(max(abs(direct-truth))),mass_refinement_max_abs=float(max(abs(half_mass-truth))),space_velocity_refinement_max_abs=float(max(abs(refined-truth))))
    assert max(checks.values())<2e-6,checks
    print('entropy r7 independent mass/basis refinements pass',flush=True)
    identity = dict(normalization=0.,current_balance=0.,first_moment_current=0.,summed_leading_second=0.,constant_local_leak=0.,density_correction_normalization=0.,field_reflection=0.,label_reversal=0.,uniform_force=0.,equal_bath=0.)
    minimum_density=np.inf; covariance_minimum=np.inf; minimum_excess=np.inf; fourth_mixture_minimum=np.inf; corners=0
    for friction,ratio,field,temp,contrast,delta,rate,fx,fy in itertools.product([.7,1.6],[.25,8.],[-5.,5.],[.8,1.4],[0.,.65],[0.,.65],[.2,2.],[-1.5,1.5],[-1.5,1.5]):
        e=ref.experiment(fx,fy,ratio,field,temp,contrast,3,delta,rate)
        state=oracle.limiting_state(e,friction);rho=state['density'];t=e['temperature']*(1+contrast*np.cos(state['phase']));derivative=state['derivative'];dx=state['dx'];second=state['second'];first=state['first'];fourth=state['fourth']
        minimum_density=min(minimum_density,float(rho.min()));minimum_excess=min(minimum_excess,state['value'])
        identity['normalization']=max(identity['normalization'],abs(rho.sum()*dx-1))
        identity['current_balance']=max(identity['current_balance'],float(max(abs(derivative@state['current_x']))))
        identity['first_moment_current']=max(identity['first_moment_current'],float(max(abs(first[1,0].sum(axis=0)-state['current_x']))),float(max(abs(first[0,1].sum(axis=0)-state['current_y']))))
        identity['summed_leading_second']=max(identity['summed_leading_second'],float(max(abs(second[2,0].sum(axis=0)-rho*t))),float(max(abs(second[0,2].sum(axis=0)-rho*t))),float(max(abs(second[1,1].sum(axis=0)))))
        local=friction*(second[2,0]/state['temperatures']-state['zeroth'][0,0])+friction*ratio*(second[0,2]/state['temperatures']-state['zeroth'][0,0])
        identity['constant_local_leak']=max(identity['constant_local_leak'],float(max(abs(local.sum(axis=0)/rho-state['leak']))))
        identity['density_correction_normalization']=max(identity['density_correction_normalization'],float(max(abs(state['density_correction'][0,0].sum(axis=1)*dx))))
        for s in range(2):
            cov=np.empty((len(rho),2,2));cov[:,0,0]=2*second[2,0][s]/rho;cov[:,1,1]=2*second[0,2][s]/rho;cov[:,0,1]=cov[:,1,0]=2*second[1,1][s]/rho
            covariance_minimum=min(covariance_minimum,float(np.linalg.eigvalsh(cov/t[:,None,None]).min()))
        fourth_mixture_minimum=min(fourth_mixture_minimum,float(min(fourth[4,0].sum(axis=0)/rho-3*t*t)),float(min(fourth[0,4].sum(axis=0)/rho-3*t*t)))
        actual=state['value']
        identity['field_reflection']=max(identity['field_reflection'],abs(actual-oracle.coefficients([dict(e,magnetic=-field,force_y=-fy)],friction)[0]))
        identity['label_reversal']=max(identity['label_reversal'],abs(actual-oracle.coefficients([dict(e,bath_contrast=-delta)],friction)[0]))
        if contrast==0:
            _,_,expected=uniform_covariance_entropy(e,friction,.001)
            identity['uniform_force']=max(identity['uniform_force'],abs(actual-expected))
        if delta==0:
            identity['equal_bath']=max(identity['equal_bath'],abs(actual-previous.coefficients([e],friction)[0]))
        corners+=1
    assert minimum_density>0 and covariance_minimum>0 and fourth_mixture_minimum>-1e-9
    assert max(identity.values())<2e-7,identity
    print('entropy r7 512 domain corners and moment identities pass',flush=True)
    closed=[]
    for friction,delta,kappa,a,k,t in [(1.1,.6,.2,.6,1,1.),(.7,.65,.3,.65,3,1.4),(1.6,.4,2.,.35,2,.8),(1.1,.6,.55,.6,1,1.),(1.1,0.,.3,.6,2,1.2)]:
        e=ref.experiment(0.,0.,1.,0.,t,a,k,delta,kappa);expected,source,factor=isotropic_gradient_formula(e,friction)
        closed.append(dict(input=e,friction=friction,expected=float(expected),source=float(source),factor=float(factor),error=float(oracle.coefficients([e],friction)[0]-expected)))
    assert max(abs(c['error']) for c in closed)<2e-7 and closed[3]['factor']==1.
    fast=[]
    for e in [inputs[0],inputs[8],inputs[10]]:
        entries=[]
        for kappa in [.2,2.,20.,200.]:
            ee=dict(e,switch_rate=kappa);value=oracle.coefficients([ee],gamma)[0];source=shortcut.coefficients([ee],gamma)[0]
            entries.append(dict(kappa=kappa,error=float(abs(value-source))))
        assert all(a['error']>b['error']>0 for a,b in zip(entries,entries[1:])) and entries[-1]['error']<2e-4
        fast.append(dict(input=e,sequence=entries))
    # A separate zero-field/no-force exact gradient anchor also checks the
    # nonGaussian fast velocity fourth moment, not merely the readout.
    e=ref.experiment(0.,0.,1.,0.,1.,.6,1,.6,.2);state=oracle.limiting_state(e,gamma);temperature=e['temperature']*(1+e['contrast']*np.cos(state['phase']))
    gaussian_mixture_fourth=3*temperature**2*(1+e['bath_contrast']**2*gamma/(gamma+e['switch_rate']))
    fourth_error=float(max(abs(state['fourth'][4,0].sum(axis=0)/state['density']-gaussian_mixture_fourth)))
    assert fourth_error<2e-7
    uniform_errors=[];scalar_errors=[];energy_errors=[];entropy_balance_errors=[];finite_covariance_minimum=np.inf
    for e in [ref.experiment(0.,0.,.4,2.2,1.,0.,1,.65,.2),ref.experiment(.2,-.15,2.,-1.8,1.2,0.,2,.6,1.2)]:
        mass=.001;expected,leak,excess=uniform_covariance_entropy(e,gamma,mass)
        uniform_errors.append(abs(ref.finite_mass_entropy(e,gamma,mass)-expected));uniform_errors.append(abs(ref.finite_mass_excess(e,gamma,mass)-excess))
    for ratio,delta,rate in [(1.,.65,.2),(.25,.6,2.),(8.,.55,.2)]:
        e=ref.experiment(0.,0.,ratio,0.,1.,0.,1,delta,rate);mass=.001
        expected=sum(drag*rate*delta**2/((drag+rate)*(1-delta**2)) for drag in [gamma,gamma*ratio])/mass
        scalar_errors.append(abs(ref.finite_mass_entropy(e,gamma,mass)-expected))
    for e in [inputs[0],inputs[8],inputs[10]]:
        mass=.002*(gamma*min(1.,e['drag_ratio']))**2/(e['temperature']*e['wavenumber']**2)
        phase,temperatures,states,basis_temp=ref.kinetic_state(e,gamma,mass,degree=16,spatial=49)
        gradients=np.array([-e['temperature']*(1+sign*e['bath_contrast'])*e['contrast']*e['wavenumber']*np.sin(phase) for sign in [1,-1]])
        energies=[basis_temp*(c[0,0]+np.sqrt(2)/2*(c[2,0]+c[0,2])) for c in states]
        moment_entropy=0.;heat=0.;work=0.
        for s,(temperature,gradient,c) in enumerate(zip(temperatures,gradients,states)):
            jx=np.sqrt(basis_temp/mass)*c[1,0];jy=np.sqrt(basis_temp/mass)*c[0,1]
            flux=basis_temp**1.5/(2*np.sqrt(mass))*(4*c[1,0]+np.sqrt(6)*c[3,0]+np.sqrt(2)*c[1,2])
            moment_entropy+=2*np.pi*np.mean((e['force_x']*jx+e['force_y']*jy)/temperature-flux*gradient/temperature**2+e['switch_rate']/mass*(energies[1-s]-energies[s])/temperature)
            work+=2*np.pi*np.mean(e['force_x']*jx+e['force_y']*jy)
            for drag,pair in [(gamma,(2,0)),(gamma*e['drag_ratio'],(0,2))]:
                heat+=2*np.pi*drag/mass*np.mean(basis_temp*(c[0,0]+np.sqrt(2)*c[pair])-temperature*c[0,0])
            cov=np.empty((len(phase),2,2));cov[:,0,0]=1+np.sqrt(2)*c[2,0]/c[0,0]-(c[1,0]/c[0,0])**2;cov[:,1,1]=1+np.sqrt(2)*c[0,2]/c[0,0]-(c[0,1]/c[0,0])**2
            cov[:,0,1]=cov[:,1,0]=c[1,1]/c[0,0]-c[1,0]*c[0,1]/c[0,0]**2
            finite_covariance_minimum=min(finite_covariance_minimum,float(np.linalg.eigvalsh(cov).min()))
        energy_errors.append(abs(heat-work));entropy_balance_errors.append(abs(moment_entropy-ref.finite_mass_entropy(e,gamma,mass,degree=16,spatial=49)))
    assert max(uniform_errors+scalar_errors+energy_errors+entropy_balance_errors)<2e-7 and finite_covariance_minimum>0
    corner=dict(ref.experiment(.15,-.18,.25,4.5,1.4,.65,3,.65,.2),detector_phase=-np.pi/2);corner_errors=[];corner_refinements=[]
    for friction in [.7,1.6]:
        base=ref.predict([corner],friction)[0];corner_errors.append(abs(base-oracle.coefficients([corner],friction)[0]));corner_refinements.append(abs(base-ref.predict([corner],friction,degree=16,spatial=49)[0]))
    assert max(corner_errors)<2e-6 and max(corner_refinements)<2e-7
    return dict(spatial_port_checks=spatial_port_checks(ref,oracle,shortcut,previous),reference_errors=checks,identities=identity,domain_corner_count=corners,minimum_position_density=minimum_density,
                minimum_excess_rate_observed=minimum_excess,minimum_leading_scaled_velocity_covariance=covariance_minimum,
                minimum_unconditional_fourth_mixture_excess=fourth_mixture_minimum,nonGaussian_fourth_anchor_error=fourth_error,
                isotropic_gradient_closed_anchors=closed,accidental_kappa_half_gamma_equality_checked=True,fast_contact_limit=fast,
                uniform_finite_mass_reference_error=max(uniform_errors),uniform_zero_field_scalar_formula_error=max(scalar_errors),
                finite_mass_first_law_error=max(energy_errors),finite_mass_entropy_energy_balance_error=max(entropy_balance_errors),
                minimum_conditional_finite_mass_velocity_covariance=finite_covariance_minimum,allowed_corner_reference_error=max(corner_errors),
                allowed_corner_refinement_error=max(corner_refinements),seconds=time.monotonic()-start)


def validate(noise_trials):
    start=time.monotonic();ref=module(TASK/'tests/reference.py','entropy_r7_ref');oracle=module(TASK/'solution/model.py','entropy_r7_oracle');shortcut=module(ROOT/'scripts/entropy_anomaly_baseline.py','entropy_r7_shortcut');previous=module(ARCHIVE/'tasks/entropy-anomaly/solution/model.py','entropy_r6_oracle')
    records=json.loads((TASK/'environment/data/calibration.json').read_text());private=json.loads((TASK/'tests/data/calibration.json').read_text())
    assert records==private and len(records)==192
    assert (TASK/'environment/data/calibration.json').read_bytes()==(ARCHIVE/'tasks/entropy-anomaly/environment/data/calibration.json').read_bytes()
    assert all(r['input']['bath_contrast']==r['input']['magnetic']==0 for r in records)
    inputs=[r['input'] for r in records];true=ref.TRUE_PARAMETER;sigma=np.array([r['sigma'] for r in records]);coefficient=oracle.coefficients(inputs);noiseless=previous.coefficients(inputs,true)
    equivalence={}
    for gamma in [.7,1.1,1.6]:
        expected=previous.coefficients(inputs,gamma)
        for label,mod in [('oracle',oracle),('shortcut',shortcut)]:equivalence[label+str(gamma)]=float(max(abs(expected-mod.coefficients(inputs,gamma))))
    assert max(equivalence.values())<2e-7
    sample_indices=[0,7,19,31,47,63,79,95];cal_ref=ref.predict([inputs[i] for i in sample_indices],true)
    cal_error=float(max(abs(cal_ref-noiseless[sample_indices])));assert cal_error<2e-6
    hidden=ref.hidden_inputs();tref=time.monotonic();truths={key:ref.predict(es,true) for key,es in hidden.items()};cold_reference_seconds=time.monotonic()-tref
    assert cold_reference_seconds<45
    controls={}
    for label,mod in [('oracle',oracle),('shortcut',shortcut)]:
        model=mod.Model().fit(records);residual=(model.predict(inputs)-np.array([r['value'] for r in records]))/sigma
        metrics=dict(parameter=model.friction,parameter_relative_error=abs(model.friction/true-1),calibration_chi2=float(residual@residual/(len(records)-1)),hidden={key:error(model.predict(es),truths[key]) for key,es in hidden.items()})
        assert metrics['parameter_relative_error']<.03 and metrics['calibration_chi2']<1.5
        for key,value in metrics['hidden'].items():assert value<.04 if key in ANCHORS or label=='oracle' else value>.04
        controls[label]=metrics
    print('entropy r7 fitted controls pass intended gates',flush=True)
    rng=np.random.default_rng(NOISE_SEED);fitted=[];noise={label:dict(max_parameter_error=0.,max_chi2=0.,hidden_min=1e10,hidden_max=0.,anchor_max=0.,successful_trials=0) for label in ['oracle','shortcut']}
    for i in range(noise_trials):
        values=noiseless+rng.normal(0,sigma);inverse=np.sum(coefficient*values/sigma**2)/np.sum(coefficient**2/sigma**2);gamma=float(1/inverse);chi=float(np.sum(((coefficient/gamma-values)/sigma)**2)/(len(values)-1));fitted.append(gamma)
        sample=[dict(input=e,value=float(v),sigma=float(s)) for e,v,s in zip(inputs,values,sigma)]
        for label,mod in [('oracle',oracle),('shortcut',shortcut)]:
            model=mod.Model().fit(sample);assert abs(model.friction-gamma)<1e-12
            scores={key:error(model.predict(es),truths[key]) for key,es in hidden.items()};diagnostic=[v for key,v in scores.items() if key not in ANCHORS];anchors=[scores[key] for key in ANCHORS]
            assert abs(gamma/true-1)<.03 and chi<1.5 and max(anchors)<.04
            assert max(diagnostic)<.04 if label=='oracle' else min(diagnostic)>.04
            row=noise[label];row['max_parameter_error']=max(row['max_parameter_error'],abs(gamma/true-1));row['max_chi2']=max(row['max_chi2'],chi);row['hidden_min']=min(row['hidden_min'],min(diagnostic));row['hidden_max']=max(row['hidden_max'],max(diagnostic));row['anchor_max']=max(row['anchor_max'],max(anchors));row['successful_trials']+=1
    print('entropy r7 all 256 noisy controls pass',flush=True)
    report=dict(revision=7,status='scientific_and_local_controls_pass_model_evaluation_pending',calibration_records=len(records),calibration_bytes_preserved_from_r6=True,calibration_copies_equal=True,calibration_equivalence=equivalence,calibration_finite_mass_reference_error=cal_error,cold_hidden_reference_seconds=cold_reference_seconds,controls=controls,noise_trials=noise_trials,noise_seed=NOISE_SEED,noise=noise,fitted_parameter_range=[min(fitted),max(fitted)],independent_checks=physics_checks(ref,oracle,shortcut,previous,truths),local_controls=local_controls())
    report['seconds']=time.monotonic()-start
    files=sorted(p for p in TASK.rglob('*') if p.is_file() and not any(part in ['__pycache__','.pytest_cache'] for part in p.parts));files += [ROOT/'scripts/entropy_anomaly_baseline.py',Path(__file__).resolve()]
    report['source_sha256']={str(p.relative_to(ROOT)):fingerprint(p) for p in files};return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--noise-trials',type=int,default=256);parser.add_argument('--output',type=Path,default=RESULTS/'entropy-anomaly-r7-validation.json');args=parser.parse_args()
    report=validate(args.noise_trials);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print('entropy r7 validation PASS',report['seconds'],flush=True)


if __name__=='__main__':main()
