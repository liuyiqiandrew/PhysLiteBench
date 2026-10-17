from pathlib import Path
import importlib.util,json,hashlib,re,time
import numpy as np
ROOT=Path('/private/tmp/physlitebench-hardening-20261005/staging/luna-high-hardening-20261005')
JOB=ROOT.parents[1]/'jobs/entropy-anomaly-r6-plain-20261006-024453'

def load(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def score(actual,truth):return float(np.linalg.norm(actual-truth)/np.linalg.norm(truth))

oracle=load(JOB/'frozen-task/solution/model.py','frozen_oracle')
ref=load(JOB/'frozen-task/tests/reference.py','frozen_independent')
source=load(ROOT/'scripts/entropy_anomaly_baseline.py','source_shortcut')
records=json.loads((JOB/'frozen-task/environment/data/calibration.json').read_text());cal=[r['input'] for r in records]
hidden=ref.hidden_inputs();flat=sum(hidden.values(),[])
summary=json.loads((ROOT/'results/entropy-anomaly-r6-results.json').read_text())
start=time.perf_counter();rng=np.random.default_rng(610661)
interiors=[]
for _ in range(64):
    interiors.append(ref.experiment(rng.uniform(-1.5,1.5),rng.uniform(-1.5,1.5),rng.uniform(.25,8),rng.uniform(-5,5),rng.uniform(.8,1.4),rng.uniform(.005,.65),int(rng.integers(1,4)),rng.uniform(.005,.65),rng.uniform(.2,2)))
# Include exactly vanishing profile/bath contrast and closed conservation anchors.
anchors=[ref.experiment(.2,-.3,ratio,B,1.1,a,3,d,k) for ratio,B,a,d,k in [(1.,0.,.6,0.,.2),(.25,-5.,.65,0.,2.),(8.,5.,0.,.65,.2),(1.,0.,0.,.65,.2),(4.,-3.,0.,.6,2.),(1.,0.,.6,.6,.55)]]
rows=[]
for name in ['6aaek3Y','Dvgx4xe','VKPRXCy']:
    directory=JOB/('entropy-anomaly__'+name);path=directory/'artifacts/app/model.py'
    collected=next(t for t in summary['trials'] if t['trial'].endswith(name))
    row={k:collected[k] for k in ['trial','reward','exception','native_contexts','metrics','source_sha256']}
    row['source_exists']=path.exists();row['native_trace_sha256']=digest(next((directory/'agent/sessions').rglob('*.jsonl')))
    if not path.exists():
        row.update(classification='timeout_with_missing_submission_after_failed_file_replacement',physical_failure=False,
                   evidence=dict(native_lines={'physics_recognition':28,'moment_expansion_commentary':174,'duplicate_delete_add_patch':175,'duplicate_patch_rejection':177,'delete_then_invalid_add':181,'successful_delete_and_invalid_add_output':184,'explicit_absence_check':188,'ls_missing_output':191},
                                 cause='The combined delete/add was rejected for targeting the same file twice. A later successful deletion was followed by an invalid Add File patch without required plus-prefixed content; the next ls confirmed model.py absent. Verification failed importing model, and the agent phase timed out.',
                                 source_state='No model.py exists in the collected app; the starter was deleted. The attempted fourth-order moment source in native tool inputs was never installed and is not treated as a final submission.',
                                 physics_recognition='The agent explicitly identified ignored switch-rate coupling, actual kappa/m scaling, and a fourth-order conditional velocity moment expansion before implementation failed.'))
        rows.append(row);continue
    m=load(path,'trial_'+name);fit=m.Model().fit(records);gamma=float(fit.friction)
    actual=fit.predict(flat);correct=oracle.coefficients(flat,gamma)
    old=source.coefficients(flat,gamma)
    rnd=interiors+anchors;pred=fit.predict(rnd);right=oracle.coefficients(rnd,gamma)
    row['fixed_fitted_friction']=gamma
    row['fixed_parameter_hidden_oracle_max_abs']=float(max(abs(actual-correct)))
    row['fixed_parameter_64_random_and_6_anchor_max_abs']=float(max(abs(pred-right)))
    row['fixed_parameter_offgrid_group_relative_error']=score(pred,right)
    row['per_hidden_group_fixed_parameter_error']={key:score(fit.predict(es),oracle.coefficients(es,gamma)) for key,es in hidden.items()}
    row['calibration_oracle_max_abs']=float(max(abs(fit.predict(cal)-oracle.coefficients(cal,gamma))))
    if name=='6aaek3Y':
        row.update(classification='clean_completed_mixed_contact_physical_shortcut_failure',physical_failure=True,
                   fixed_parameter_shortcut_equivalence_max_abs=float(max(abs(actual-old))),
                   offgrid_shortcut_equivalence_max_abs=float(max(abs(pred-source.coefficients(rnd,gamma)))),
                   trace_evidence=dict(native_trace='The actual final source retains the original prediction formula, which divides the exact single-contact entropy by 1-bath_contrast^2 and ignores switch_rate in the entropy readout. Only fit and input validation were completed. Public tests passed; no exception occurred.',
                                       api='fit returns self, fitted friction and calibration pass; all three contact diagnostics fail while both equal-contact and uniform-force anchors pass.'))
        # Preserve the exact final fitting implementation and add only the
        # physical kinetic-contact prediction correction. Test without refit.
        original=path.read_text();helper=(JOB/'frozen-task/solution/model.py').read_text()
        helper=helper[helper.index('def limiting_state('):helper.index('@lru_cache(4096)\ndef entropy_rate(')]
        repaired=original.replace('def entropy_rate(', 'def mixed_contact_entropy_rate(',1)
        helper=helper.replace('def limiting_state(', 'def finite_contact_state(',1)
        addition='''\n@lru_cache(4096)\ndef entropy_rate(force_x,force_y,drag_ratio,magnetic,temperature,contrast,wavenumber,bath_contrast,switch_rate,friction):\n    if bath_contrast==0. or contrast==0.:\n        return mixed_contact_entropy_rate(force_x,force_y,drag_ratio,magnetic,temperature,contrast,wavenumber,bath_contrast,switch_rate,friction)\n    experiment=dict(zip(INPUT_FIELDS,(force_x,force_y,drag_ratio,magnetic,temperature,contrast,wavenumber,bath_contrast,switch_rate)))\n    return finite_contact_state(experiment,friction)['value']\n'''
        repair_path=ROOT/'results/entropy-anomaly-r6-6aaek3Y-fixed-fast-contact.py'
        repair_path.write_text(repaired+'\n\n'+helper+addition)
        fixed=load(repair_path,'causal_fast_contact_fix');f=fixed.Model();f.friction=gamma
        repaired_hidden=f.predict(flat);repaired_random=f.predict(rnd)
        row['causal_physical_repair']=dict(path=str(repair_path.relative_to(ROOT)),sha256=digest(repair_path),
            description='Retain the exact submission fit/interface. Replace only the mixed-contact prediction on nonzero thermal/contact contrast by the finite-kappa/m velocity/contact moment hierarchy; no refit or calibration/data change.',
            fitted_friction_unchanged=f.friction==gamma,hidden_max_abs=float(max(abs(repaired_hidden-correct))),
            offgrid_max_abs=float(max(abs(repaired_random-right))),calibration_change_max_abs=float(max(abs(f.predict(cal)-fit.predict(cal)))),
            hidden_error_at_original_true_parameter={key:score(f.predict(es),oracle.coefficients(es,1.1)) for key,es in hidden.items()})
        assert row['fixed_parameter_shortcut_equivalence_max_abs']<1e-12 and row['causal_physical_repair']['hidden_max_abs']<1e-10
        assert row['causal_physical_repair']['offgrid_max_abs']<1e-10 and row['causal_physical_repair']['calibration_change_max_abs']<1e-12
        assert max(row['causal_physical_repair']['hidden_error_at_original_true_parameter'].values())<.04
    else:
        row.update(classification='reward_pass_with_agent_timeout_and_correct_integrated_kinetic_contact_observable',physical_failure=False,
                   trace_evidence=dict(first_fit_patch_line=88,physics_recognition_commentary_line=360,moment_implementation_line=361,permutation_symmetrization_fix_line=389,random_interior_check_before_timeout_line=410,
                                       interpretation='The final installed model derives conditional second/fourth leading moments and first/third corrections for the fast kinetic-contact process. It retains the correct calibrated single-contact/uniform-profile reductions. Agent timeout remains an exception and is not erased by the verifier pass.'),
                   omitted_label_density_correction='This tensor implementation omits the O(m) antisymmetric conditional density and its associated thermal source. Their contribution to the spatially integrated observable vanishes: each conditional correction is a periodic derivative with zero integral, and the baths are proportional to the same T(x). The fast relaxation matrix and kappa are spatially constant, so after division by T_s(x) these terms have constant coefficients multiplying those zero integrals. This is an equivalent integrated observable, not a physical error.',
                   observable_exactness='Agreement of the integrated entropy with the complete raw-moment oracle on every hidden preparation, 64 random allowed interiors and six exact anchors is reported below; the final code is not claimed to reconstruct all local conditional density corrections.')
        # Check omitted-label-density scalar contribution explicitly at each
        # hidden point using the complete oracle hierarchy's alternate source.
        label_integrals=[]
        for e in flat:
            state=oracle.limiting_state(e,gamma)
            label_integrals.extend(np.sum(state['density_correction'][0,0],axis=1)*state['dx'])
        row['omitted_conditional_density_integral_max_abs']=float(max(abs(np.array(label_integrals))))
        independent=[]
        for e in [flat[0],flat[4],flat[8],interiors[0],interiors[1],anchors[2]]:
            predicted=fit.predict([e])[0];finite=ref.predict([e],gamma)[0]
            independent.append(dict(input=e,submitted=float(predicted),finite_mass_paired_reference=float(finite),absolute_error=float(abs(predicted-finite))))
        row['independent_finite_mass_checks']=independent
        row['independent_finite_mass_max_abs']=max(c['absolute_error'] for c in independent)
        assert row['fixed_parameter_hidden_oracle_max_abs']<1e-10 and row['fixed_parameter_64_random_and_6_anchor_max_abs']<1e-10
        assert row['independent_finite_mass_max_abs']<1e-6
    rows.append(row)
report=dict(task='entropy-anomaly',revision=6,job=JOB.name,model='gpt-5.6-luna',effort='high',hint=False,
    reward_passes=1,agent_timeout_exceptions=2,clean_intended_physical_failures=1,missing_submission_implementation_failures=1,
    qualification='Does not meet zero-of-three target: one actual reward pass is preserved, and two timeouts cannot be reclassified as clean physical failures.',
    source_matches_frozen_plan=summary['source_matches_frozen_plan'],task_checksum=summary['trials'][0]['task_checksum'],
    fixed_parameter_review_seed=610661,random_interior_count=64,extra_anchor_count=6,trials=rows,seconds=time.perf_counter()-start,
    causal_repair_scope='Local scientific comparison of the preserved final source at its original fitted friction. No model run, trial resampling, calibration alteration, hidden-test change or refit.',
    immutable_frozen_source_sha256={str(p.relative_to(JOB/'frozen-task')):digest(p) for p in sorted((JOB/'frozen-task').rglob('*')) if p.is_file()})
path=ROOT/'results/entropy-anomaly-r6-trial-review.json';path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['trials','immutable_frozen_source_sha256']},indent=2))
print(json.dumps([{k:t[k] for k in ['trial','classification','fixed_parameter_hidden_oracle_max_abs','fixed_parameter_64_random_and_6_anchor_max_abs','independent_finite_mass_max_abs'] if k in t} for t in rows],indent=2))
