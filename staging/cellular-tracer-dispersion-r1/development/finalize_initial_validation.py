"""Finalize completed science after a test-summary count correction; no science rerun."""
from pathlib import Path
import hashlib,json
BASE=Path(__file__).resolve().parents[1];TASK=BASE/'tasks/cellular-tracer-dispersion'
report=json.loads((BASE/'results/author-check-failure-1791181754232509000.json').read_text())
assert report['status']=='author_check_failed'
assert report['traceback'].endswith('AssertionError\n') and "'8 passed'" in report['traceback']
controls=json.loads((BASE/'results/cellular-tracer-dispersion-r1-local-controls.json').read_text())
assert controls['oracle']['returncode']==0 and '7 passed' in controls['oracle']['stdout']
assert controls['shortcut']['returncode']==1 and '3 failed, 4 passed' in controls['shortcut']['stdout']
for row in report['noise']['rows']:
    for name in ['oracle','shortcut']:
        r=row[name];assert r['chi2']<1.5 and r['parameter_error']<.03
        assert all(v<.04 if name=='oracle' or k=='anchors' else v>.04 for k,v in r['hidden'].items())
assert len(report['noise']['rows'])==256 and report['domain']['maximum_spectral_refinement']<1e-7
assert report['domain']['maximum_reference_error']<5e-4 and report['domain']['maximum_refined_reference_error']<4e-5
assert report['domain']['minimum_enhancement_eigenvalue']>-1e-10
assert report['domain']['maximum_fixed_point_residual']<1e-10 and report['domain']['maximum_symmetry_error']<1e-9
assert max(r['relative_time_error'] for r in report['dynamic_checks']['finite_time_displacement_moments'])<5e-4
assert report['dynamic_checks']['counting_eigenvalue_curvature']['relative_error']<1e-5
report.pop('traceback')
report['status']='science_and_local_controls_complete'
report['local_controls']={k:{x:y for x,y in v.items() if x not in ['stdout','stderr']} for k,v in controls.items()}
report['source_sha256']={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TASK/'environment/model.py',TASK/'solution/model.py',TASK/'tests/reference.py',BASE/'scripts/cellular_tracer_dispersion_baseline.py',BASE/'scripts/validate_cellular_tracer_dispersion.py']}
report['execution_history']=json.loads((BASE/'development/local-count-correction.json').read_text())
report['execution_history']['duration_note']='Full author-run elapsed time was not saved before the bookkeeping assertion. Both actual local control durations are retained.'
path=BASE/'results/cellular-tracer-dispersion-r1-validation.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'status':report['status'],'noise':{k:v for k,v in report['noise'].items() if k!='rows'},'domain':{k:v for k,v in report['domain'].items() if k!='rows'},'actual':report['actual_data']},indent=2))
