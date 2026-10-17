"""Finalize immutable r5 trial diagnosis and verified full archive."""
from pathlib import Path
import json,hashlib,shutil,ast
ROOT=Path('/private/tmp/physlitebench-hardening-20261005');STAGE=ROOT/'staging/luna-high-hardening-20261005';JOB=ROOT/'jobs/superconducting-heat-r5-plain-20261006-035908';OUT=STAGE/'results/prototypes/superconducting-heat-r5-hydro-review'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
causal=json.loads((OUT/'causal-review.json').read_text());reportpath=STAGE/'results/superconducting-heat-r5-trial-reviews.json';report=json.loads(reportpath.read_text());shutil.copy2(reportpath,OUT/'root-initial-trial-reviews.json')
for name,row in report['trials'].items():
    if row['reward']==1:
        row['classification']='physical_pass'
        row['classification_scope']='Full correct oscillator energy/coherence covariance, all spin feedback and calorimeter channels, by selecting the exact branch supplied in the public r5 scaffold. No claim of an independent-from-scratch derivation.'
        assert row['independent_maximum_error_pW']<1e-7 and max(row['same_parameter_hidden_relative_errors'].values())<.04
    else:
        assert name==causal['failed_trial']
        row['classification']='physical_model_failure';row['error_subtype']='static_opposite_spin_pair_relative_phase';row['classification_scope']=causal['classification_scope'];row['single_effect_causal_repair']={k:causal[k] for k in ['native_conductance','conductance_unchanged','repair','hamiltonian_derivation','hidden','calibration_prediction_change','offgrid_repaired_maximum_error_pW','offgrid_original_maximum_error_pW','source_sha256','diagnostic_sha256']}
    row['phase_shift_allegation']=causal['phase_shift']
    row['native_source_provenance']=next(x for x in causal['native_provenance'] if x['trial']==name)
report.update(reviewer='root quantitative review plus independent hydro physics/causal audit',clean_physical_failures=1,physical_passes=2,mathematical_or_coding_failures=0,infrastructure_failures=0,independent_causal_report=str((OUT/'causal-review.json').relative_to(STAGE)),independent_causal_report_sha256=sha(OUT/'causal-review.json'),qualifies_zero_of_three=False,qualification='Does not meet0/3: two correct passes. Failed third artifact repaired the targeted factorization, then introduced a distinct physical pair-relative-phase error. The known exact-branch public scaffold issue independently prevents transfer of this evidence to the corrected public revision.',all_original_submissions_unchanged=True)
reportpath.write_text(json.dumps(report,indent=2)+'\n')
for source in ['/private/tmp/superconducting-r5-causal-review.py','/private/tmp/superconducting-r5-causal-review.log','/private/tmp/finalize_superconducting_r5_review_archive.py']:shutil.copy2(source,OUT/Path(source).name)
plan=json.loads((STAGE/'results/superconducting-heat-r5-evaluation-plan.json').read_text());frozen_checks=[]
for relative,expected in plan['source_sha256'].items():
    p=JOB/'frozen-task'/relative;assert sha(p)==expected
    active=STAGE/'tasks/superconducting-heat'/relative;assert sha(active)==expected
    frozen_checks.append(dict(path=relative,sha256=expected,active_equals_frozen=True))
archive=STAGE/'archives/superconducting-heat-r5';assert not archive.exists(),'Preserve existing archive instead of replacing it'
archive.mkdir();copies=[]
def copy_file(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);assert sha(source)==sha(dest);copies.append(dict(path=str(dest.relative_to(archive)),sha256=sha(dest),bytes=dest.stat().st_size,source=str(source)))
def copy_tree(source,dest):
    for p in sorted(source.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts:copy_file(p,dest/p.relative_to(source))
copy_tree(STAGE/'tasks/superconducting-heat',archive/'tasks/superconducting-heat')
for name in ['superconducting_heat_baseline.py','validate_superconducting_heat.py','review_superconducting_r5.py','run_science.py','freeze_evaluation.py','collect_evaluation.py']:
    p=STAGE/'scripts'/name
    if p.exists():copy_file(p,archive/'scripts'/name)
for p in sorted((STAGE/'results').glob('superconducting-heat-r5*')):
    if p.is_file():copy_file(p,archive/'results'/p.name)
copy_tree(OUT,archive/'results/prototypes'/OUT.name)
brownian=STAGE/'results/prototypes/superconducting-heat-r5-brownian-audit'
if brownian.exists():copy_tree(brownian,archive/'results/prototypes'/brownian.name)
copy_tree(STAGE/'superconducting-heat/phase-oscillator-prototype',archive/'superconducting-heat/phase-oscillator-prototype')
# Keep the runnable author validator/prototype's exact archived r4 dependency
# in the same relative layout; preserve its complete provenance and raw records.
copy_tree(STAGE/'archives/superconducting-heat-r4',archive/'archives/superconducting-heat-r4')
for job in sorted((ROOT/'jobs').glob('superconducting-heat-r5-*')):copy_tree(job,archive/'jobs'/job.name)
for name in ['AGENTS.md','docs/ADDING_TASKS.md','docs/BENCHMARK.md']:
    p=STAGE/name
    if p.exists():copy_file(p,archive/name)
manifest=dict(task='superconducting-heat',revision=5,review_sha256=sha(reportpath),causal_sha256=sha(OUT/'causal-review.json'),frozen_task_checks=frozen_checks,contains_complete_paid_batch=True,contains_complete_oracle_and_shortcut_jobs=True,runnable_dependency='archives/superconducting-heat-r4 preserved at its original relative position',file_count=len(copies),total_bytes=sum(x['bytes'] for x in copies),files=copies)
(archive/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for row in copies:assert sha(archive/row['path'])==row['sha256']
# Originals still agree with collection and native provenance after archival.
for row in causal['native_provenance']:
    f=JOB/row['trial']
    for item in row['files']:assert sha(f/item['path'])==item['sha256']
print(json.dumps(dict(review=str(reportpath),review_sha256=sha(reportpath),archive=str(archive),manifest_sha256=sha(archive/'manifest.json'),files=len(copies),bytes=manifest['total_bytes'],classes={name:row['classification'] for name,row in report['trials'].items()},causal_repaired_maximum_offgrid_pW=causal['offgrid_repaired_maximum_error_pW'],calibration_change=causal['calibration_prediction_change']),indent=2))
