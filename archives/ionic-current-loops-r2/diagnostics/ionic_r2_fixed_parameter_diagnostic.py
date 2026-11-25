from pathlib import Path
import hashlib, importlib.util, json, types
import numpy as np

root=Path('/Users/andrewliu/Library/CloudStorage/OneDrive-PrincetonUniversity/Courses/AI Agent/project1/PhysLiteBench')
review_path=root/'results/zero-three-ionic-r2-partial-review.json'
review=json.loads(review_path.read_text())
task=root/'staging/ionic-current-loops-r2/tasks/ionic-current-loops'
reference_path=task/'tests/reference.py'
spec=importlib.util.spec_from_file_location('independent_reference',reference_path)
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
old='    electric=diffusion_charge_gradient/conductivity\n'
new=('    current=-np.mean(diffusion_charge_gradient/conductivity)/np.mean(1/conductivity)\n'
     '    electric=(diffusion_charge_gradient+current)/conductivity\n')
records=json.loads((task/'tests/data/calibration.json').read_text())
inputs=[r['input'] for r in records];values=np.array([r['value'] for r in records]);sigmas=np.array([r['sigma'] for r in records])
hidden=reference.hidden_inputs();truth={key:reference.predict(experiments) for key,experiments in hidden.items()}
rows={}
for trial,reviewed in review['trials'].items():
    if reviewed['classification']!='physical_model_failure':continue
    evidence=root/reviewed['evidence_directory']
    source_path=evidence/'artifacts/app/model.py';source=source_path.read_text()
    original_hash=sha(source_path)
    assert original_hash==reviewed['source_sha256']
    assert source.count(old)==1
    fixed=source.replace(old,new)
    namespaces=[]
    for label,text in [('original',source),('repair',fixed)]:
        module=types.ModuleType(trial+'_'+label)
        exec(compile(text,str(source_path)+':'+label,'exec'),module.__dict__)
        namespaces.append(module)
    original,repaired=namespaces
    parameter=float(reviewed['metrics']['parameter'])
    # Reproduce the submitted fit once, but never fit the repaired copy.
    fitted=original.Model().fit(records)
    assert abs(fitted.diffusivity-parameter)<1e-12, (fitted.diffusivity, parameter)
    models=[original.Model(),repaired.Model()]
    for model in models:model.diffusivity=parameter
    metrics=[];predictions=[];field_checks=[]
    for module,model in zip(namespaces,models):
        calibration=model.predict(inputs)
        hp={key:model.predict(experiments) for key,experiments in hidden.items()}
        errors={key:float(np.sqrt(np.mean((hp[key]-target)**2))/np.sqrt(np.mean(target**2))) for key,target in truth.items()}
        chi2=float(np.sum(((calibration-values)/sigmas)**2)/(len(records)-1))
        metrics.append({'diffusivity':model.diffusivity,'parameter_relative_error':abs(model.diffusivity/reference.TRUE_PARAMETER-1),'calibration_chi2':chi2,'hidden':errors,'all_grader_thresholds_pass':bool(chi2<1.5 and abs(model.diffusivity/reference.TRUE_PARAMETER-1)<.03 and max(errors.values())<.04)})
        predictions.append({'calibration':calibration,'hidden':hp})
        loops=[];current_variation=[];current_means=[]
        for experiments in hidden.values():
            for e in experiments:
                key=json.dumps({k:e[k] for k in ['means','amplitudes','waves','phases']},sort_keys=True)
                x,c,electric,flux,change=module.fields(key)
                charge_current=parameter*(module.CHARGES@flux)
                loops.append(float(2*np.pi*np.mean(electric)))
                current_means.append(float(np.mean(charge_current)))
                current_variation.append(float(np.max(np.abs(charge_current-charge_current.mean()))))
        field_checks.append({'maximum_absolute_loop_emf':max(map(abs,loops)), 'maximum_charge_current_nonuniformity':max(current_variation),'maximum_absolute_uniform_charge_current':max(map(abs,current_means))})
    old_metrics=reviewed['metrics']
    reproduction=max(abs(metrics[0]['hidden'][key]-old_metrics['hidden'][key]) for key in hidden)
    assert reproduction<1e-12
    assert metrics[1]['all_grader_thresholds_pass']
    assert sha(source_path)==original_hash
    rows[trial]={
        'classification':'physical_model_failure_supported_by_fixed_parameter_repair',
        'source':str(source_path.relative_to(root)), 'source_sha256':original_hash,
        'result_sha256':sha(evidence/'result.json'), 'verifier_metrics_sha256':sha(evidence/'verifier/metrics.json'),
        'reviewed_trajectory_sha256':reviewed['trajectory_sha256'],
        'reviewed_native_transcript_sha256':reviewed['native_transcript_sha256'],
        'repair_source_sha256_in_memory':hashlib.sha256(fixed.encode()).hexdigest(),
        'parameter_held_fixed':parameter,'refit_repaired_model':False,
        'original_fit_reproduction_absolute_error':abs(fitted.diffusivity-parameter),
        'before':metrics[0],'after':metrics[1],
        'before_field_checks':field_checks[0],'after_field_checks':field_checks[1],
        'original_metrics_reproduction_absolute_error_max':reproduction,
        'calibration_prediction_change_absolute_max':float(np.max(np.abs(predictions[1]['calibration']-predictions[0]['calibration']))),
        'hidden_predictions':{key:{'reference':truth[key].tolist(),'before':predictions[0]['hidden'][key].tolist(),'after':predictions[1]['hidden'][key].tolist()} for key in hidden},
        'original_source_unchanged_after_diagnostic':True,
        'causal_interpretation':'Changing only the common electrostatic loop-current closure at the submitted fitted diffusivity restores all original grading thresholds. The final source retained a physical zero-current approximation; the prior full-trajectory review found no implementation regression from a correct common-current model.'
    }
report={
    'task':'ionic-current-loops','revision':2,'status':'completed_fixed_parameter_causal_diagnostic',
    'scope':'Two completed physical failures from the partial batch only. This is a numerical posthoc diagnostic, not a model trial, retry, new fit or change to retained artifacts.',
    'prior_review':{'path':str(review_path.relative_to(root)),'sha256':sha(review_path)},
    'physical_derivation':[
      'Nernst-Planck charge current divided by common D is j=−gprime+sigma*E, where gprime=sum_i ratio_i*z_i*dc_i/dx and sigma=sum_i ratio_i*z_i^2*c_i.',
      'Electroneutral charge conservation requires spatially constant j, not j=0.',
      'A periodic electrostatic potential and no inductive EMF require integral E dx=0.',
      'Consequently j=−mean(gprime/sigma)/mean(1/sigma) and E=(gprime+j)/sigma. The original zero-current closure need not meet the loop constraint.'
    ],
    'exact_in_memory_change':{'removed':old,'inserted':new,'number_of_replacements_per_source':1,'all_other_code_unchanged':True},
    'independent_reference':{'path':str(reference_path.relative_to(root)),'sha256':sha(reference_path),'method':'Conservative face fluxes with a periodic scalar-potential sparse solve, 512/1024 Richardson extrapolation as used by original grading.'},
    'hidden_test_sha256':sha(task/'tests/test_hidden.py'),
    'calibration_sha256':sha(task/'tests/data/calibration.json'),
    'thresholds':{'calibration_chi2':1.5,'parameter_relative_error':.03,'hidden_relative_rms':.04},
    'diagnostic_script':{'path':'/private/tmp/ionic_r2_fixed_parameter_diagnostic.py','sha256':sha(Path(__file__))},
    'trials':rows,
    'qualification':'Both reviewed completed failures have a causal physical repair. The unstarted infrastructure attempt is not a model outcome; this report does not assert three-trial qualification.'
}
out=root/'results/zero-three-ionic-r2-causal-diagnostic.json'
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'report':str(out.relative_to(root)),'sha256':sha(out),'trials':{t:{'before':r['before'],'after':r['after'],'loop_emf_before':r['before_field_checks']['maximum_absolute_loop_emf'],'loop_emf_after':r['after_field_checks']['maximum_absolute_loop_emf']} for t,r in rows.items()}},indent=2))
