from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,zipfile
W=Path('/Users/yp9846/Documents/ChatGPT/phys-lite-bench');B=W/'.presentation-failures-20261015';R=Path('/Users/yp9846/Desktop/PhysLiteBench');O=R/'output/presentation-20261014'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=O/'PhysLiteBench-10-minute-focused.pptx';assert sha(old)==sha(B/'source.pptx'),'New user save needs review'
F=W/'output/physlitebench-presentation/PhysLiteBench-10-minute-physics-failures.pptx';A=json.loads((B/'audit.json').read_text());assert sha(F)==A['final_sha256']
rev=R/'results/presentation-20261014/physics-failures-20261015';rev.mkdir(parents=True,exist_ok=True)
for name in ['source.pptx','build.mjs','finalize.mjs','changes.json','validation.json','finalization-result.json','audit.py','audit.json','remote-source.json','source-manifest.json','deliver.py']:
 shutil.copy2(B/name,rev/name)
shutil.copytree(B/'final-previews',rev/'final-previews',dirs_exist_ok=True)
out=O/F.name;assert not out.exists();shutil.copy2(F,out)
U=W/'output/PhysLiteBench-physics-failure-figures-20261015';dest=O/U.name;assert not dest.exists();shutil.copytree(U,dest)
zp=W/'output'/(U.name+'.zip')
with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(U.rglob('*')):
  if p.is_file():z.write(p,Path(U.name)/p.relative_to(U))
zo=O/zp.name;shutil.copy2(zp,zo);assert sha(F)==sha(out) and sha(zp)==sha(zo)
p=R/'results/presentation-20261014/update-on-completion.json';shutil.copy2(p,rev/'previous-update-on-completion.json');c=json.loads(p.read_text());now=datetime.now(timezone.utc).isoformat()
c.update(current_deck=str(out),current_deck_sha256=sha(out),prior_deck=str(old),prior_deck_sha256=sha(old),updated_utc=now,authoring_script=str(rev/'build.mjs'),authoring_data=str(dest/'failure-breakdown.json'),authoring_backup=str(rev),final_filename_suggestion=out.name,state='complete',backup_slides=list(range(13,20)))
c['user_request']+=' Latest request: use the published outcome-breakdown report to update physics-failure analysis, without adding or double counting attempts. Group by exact model, effort and hint, not runner identity. Highlight reviewed physics failures while separating mathematical/coding failures and execution errors.'
c['completion_condition']='All 351 scheduled observations and final audit were completed and delivered previously. Current revisions edit the completed presentation only and do not authorize additional model runs.'
c['narrative']='Title 1; benchmark 2; bead setup 3; prompt conditions 4; model/effort inventory 5; no-hint frontier 6; three-condition frontiers 7; single-matrix Luna effort comparison 8; strict reviewed physics-failure breakdown 9; bead trajectory 10; conclusions 11; Q&A 12; no-hint results 13; full-study hint results 14; roster 15; bead formula 16; no-hint efforts 17; reviewed physics failures by model/effort 18; outcome definitions/counts 19.'
c['final_update_requirements'][0]='Import the latest current deck before further edits and preserve new user changes. Keep Times New Roman, committee style, title plus ten content slides, Q&A and seven backups.'
c['final_update_requirements'].append('Failure slides 9, 18 and 19 use the strict trajectory-reviewed outcome classification for 1,170 existing matrix attempts at pinned commit f1164fe2b2bc8d312733bedbfc633b48e3b3cb36. Do not add these counts to the 2,068 unique study attempts, pool incompatible old review categories, or impute physics classifications to unreviewed attempts. Slide 8 retains 39 per cell. Runtime and historical-control limits remain in notes. The underlying published classification ledger was not independently reviewed locally.')
c['physics_failure_revision']={'completed_utc':now,'source_sha256':sha(old),'report_commit':'f1164fe2b2bc8d312733bedbfc633b48e3b3cb36','report_sha256':'1ea07bd98b76b25cb5927e6ca8fbd04defccbe3ebc5cab957ada0256b0a79309','existing_attempts_classified':1170,'baseline_identity_matches':390,'hinted_aggregate_matches':780,'new_trials_added':0,'full_study_attempts':2068,'physics_failures_by_condition':[296,158,66],'luna_high_physics_failures':[33,12,2],'attempts_per_cell':39,'changed_main_slides':[9,11],'new_backup_slides':[18,19],'data_bundle':str(dest),'data_bundle_zip':str(zo),'validation':str(rev/'validation.json'),'review':str(rev/'audit.json'),'underlying_raw_trajectories_independently_reviewed':False,'watcher_unchanged':'PAUSED'}
c['completion']['presentation_review']=str(rev/'audit.json');c['completion']['presentation_validation']=str(rev/'validation.json');c['completion']['delivery_state']='physics_failure_revision_saved_for_delivery_in_this_chat'
p.write_text(json.dumps(c,indent=2)+'\n')
receipt={'presentation':str(out),'sha256':sha(out),'bundle_zip':str(zo),'prior_user_source_preserved':str(old),'revision':str(rev)};(B/'delivery.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
