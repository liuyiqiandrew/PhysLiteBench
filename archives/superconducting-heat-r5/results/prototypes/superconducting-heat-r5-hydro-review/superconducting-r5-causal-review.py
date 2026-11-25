"""Read-only causal diagnostic of the one failed r5 final submission."""
import ast,importlib.util,json,pathlib,hashlib,time,sys
sys.dont_write_bytecode=True
import numpy as np
ROOT=pathlib.Path('/private/tmp/physlitebench-hardening-20261005');STAGE=ROOT/'staging/luna-high-hardening-20261005';JOB=ROOT/'jobs/superconducting-heat-r5-plain-20261006-035908'
OUT=STAGE/'results/prototypes/superconducting-heat-r5-hydro-review';OUT.mkdir(exist_ok=True)
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
start=time.perf_counter();rootreport=json.loads((STAGE/'results/superconducting-heat-r5-trial-reviews.json').read_text());collected=json.loads((STAGE/'results/superconducting-heat-r5-results.json').read_text())
name='superconducting-heat__A4CRALo';folder=JOB/name;source_path=folder/'artifacts/app/model.py';source=source_path.read_text();assert source.count('                sign = -parity\n')==1
patched_source=source.replace('                sign = -parity\n','                sign = (1 if n == 0 else -1)*parity\n')
patched_path=OUT/'A4CRALo-pair-sign-only-diagnostic.py';patched_path.write_text(patched_source)
original=load(source_path,'original_r5');patched=load(patched_path,'patched_r5');ref=load(JOB/'frozen-task/tests/reference.py','frozen_independent_r5')
conductance=rootreport['trials'][name]['native_conductance'];original_model=original.Model();original_model.conductance=conductance;patched_model=patched.Model();patched_model.conductance=conductance
cases=rootreport['offgrid_inputs'];original_values=original_model.predict(cases);fixed=patched_model.predict(cases);truth=ref.predict(cases,conductance)
max_ref_error=float(np.max(abs(fixed-truth)));assert max_ref_error<1e-7
hidden={}
for group,inputs in ref.hidden_inputs().items():
    independent=ref.predict(inputs,conductance);target=ref.predict(inputs,ref.TRUE_PARAMETER)
    before=original_model.predict(inputs);after=patched_model.predict(inputs)
    hidden[group]=dict(original_same_parameter_error=float(np.linalg.norm(before-independent)/np.linalg.norm(independent)),repaired_same_parameter_error=float(np.linalg.norm(after-independent)/np.linalg.norm(independent)),repaired_actual_grading_error=float(np.linalg.norm(after-target)/np.linalg.norm(target)),original=before.tolist(),repaired=after.tolist(),independent_at_native_parameter=independent.tolist())
    assert hidden[group]['repaired_same_parameter_error']<1e-7 and hidden[group]['repaired_actual_grading_error']<.04
    print(group,hidden[group]['original_same_parameter_error'],hidden[group]['repaired_same_parameter_error'],hidden[group]['repaired_actual_grading_error'],flush=True)
records=json.loads((JOB/'frozen-task/tests/data/calibration.json').read_text());inputs=[r['input'] for r in records]
cal_difference=float(np.max(abs(original_model.predict(inputs)-patched_model.predict(inputs))));assert cal_difference==0.
# Literal Hamiltonian spin matrices determine the relative amplitudes.
S=np.array([[0.,1.],[-1.,0.]],complex);table={0:np.eye(2,dtype=complex),1:np.array([[0.,1.],[0.,0.]],complex),-1:np.array([[0.,0.],[1.,0.]],complex)}
spin_proof=[]
for n in [-1,0,1]:
    electron=table[n];hole=S@table[-n].conj()@S.conj().T;expected_hole=(1 if n==0 else -1)*electron
    assert np.max(abs(hole-expected_hole))==0
    spin_proof.append(dict(harmonic=n,hole_relative_to_electron=1 if n==0 else -1,pair_anomalous_sign=1 if n==0 else -1,transfer_anomalous_sign=-1 if n==0 else 1))
# Explicit conditional oscillator covariance, independently from Fock elements.
fock=[]
for rho,tph in [(.3,12.),(.7,24.),(1.,24.)]:
    values=ref.fock_covariances(rho,tph,count=6,states=96)
    fock.append(dict(rho=rho,temperature=tph,max_cross_parity_error=max(abs(cross-(-1)**ell*pe) for ell,pe,ph,cross in values)))
    assert fock[-1]['max_cross_parity_error']<1e-14
phase_grid=np.linspace(-np.pi,np.pi,101);wrapped=(phase_grid+2*np.pi)%(2*np.pi)-np.pi
phase_equivalence=float(np.max(abs(np.exp(1j*wrapped)-np.exp(1j*(phase_grid+np.pi)))));assert phase_equivalence<1e-14
native=[]
for trial in collected['trials']:
    f=JOB/trial['trial'];rollouts=list((f/'agent/sessions').rglob('*.jsonl'));contexts=[];complete=0
    for path in rollouts:
        for line in path.read_text().splitlines():
            event=json.loads(line);payload=event.get('payload',{})
            if event.get('type')=='turn_context':contexts.append(dict(model=payload.get('model'),effort=payload.get('effort')))
            if event.get('type')=='event_msg' and payload.get('type')=='task_complete':complete+=1
    assert contexts and all(c['model']=='gpt-5.6-luna' and c['effort']=='high' for c in contexts)
    assert complete>=1 and trial['exception'] is None
    native.append(dict(trial=trial['trial'],native_contexts=contexts,native_task_complete_events=complete,files=[dict(path=str(p.relative_to(f)),sha256=sha(p)) for p in [f/'artifacts/app/model.py',f/'agent/trajectory.json',f/'result.json',f/'config.json',*rollouts]]))
report=dict(revision=5,job=JOB.name,failed_trial=name,classification='physical_model_failure_pair_relative_phase',classification_scope='The model repairs the intended factorized oscillator covariance by selecting the supplied exact branch. It then deliberately imposes the driven equal-spin pair relative phase on static opposite-spin oscillator-assisted pair creation, contrary to the singlet Hamiltonian. This is an introduced physical coherence-kernel error, not failure to repair the targeted factorization.',native_conductance=conductance,conductance_unchanged=True,repair='Replace only sign=-parity by sign=(1 if n==0 else -1)*parity in oscillator pair creation. No fit, quadrature, weights, transfer rates, spin balance, phase wrapping, or gates changed.',hamiltonian_derivation=dict(singlet_matrix='S=[[0,1],[-1,0]]',electron='T_n',hole='S T_-n* S† = T_0 for n0 and -T_n for n±1',pair_amplitude='(u_L*v_R*T_n+v_L*u_R*exp(i phi)*hole) S',oscillator='Cross covariance C_ell=(-1)^ell P_ell',consequence='Static pair kernel has +C_ell cos(phi); driven pair kernel has -C_ell cos(phi).'),spin_matrix_checks=spin_proof,fock_covariance_checks=fock,hidden=hidden,calibration_prediction_change=cal_difference,offgrid_inputs=cases,offgrid_original=original_values.tolist(),offgrid_repaired=fixed.tolist(),offgrid_independent=truth.tolist(),offgrid_repaired_maximum_error_pW=max_ref_error,offgrid_original_maximum_error_pW=float(np.max(abs(original_values-truth))),phase_shift=dict(original_was_already_correct=True,max_modulo_two_pi_error=phase_equivalence,note='All3 agents alleged a phase-shift bug; the wrapped original formula already implements phi+pi modulo2pi and their code preserves it. This commentary is not evidence of a physical repair.'),native_provenance=native,source_sha256=sha(source_path),diagnostic_sha256=sha(patched_path),root_review_sha256_before_finalization=sha(STAGE/'results/superconducting-heat-r5-trial-reviews.json'),script_sha256=sha(pathlib.Path(__file__)),seconds=time.perf_counter()-start)
(OUT/'causal-review.json').write_text(json.dumps(report,indent=2)+'\n');print('DONE',max_ref_error,report['seconds'],flush=True)
