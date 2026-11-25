from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,zipfile
W=Path('/Users/yp9846/Documents/ChatGPT/phys-lite-bench');B=W/'.presentation-costs-20261015';R=Path('/Users/yp9846/Desktop/PhysLiteBench');O=R/'output/presentation-20261014'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=O/'PhysLiteBench-10-minute-physics-failures.pptx';assert sha(old)==sha(B/'source.pptx'),'New user save needs review'
F=W/'output/physlitebench-presentation/PhysLiteBench-10-minute-all-costs-reviewed.pptx';A=json.loads((B/'audit.json').read_text());assert sha(F)==A['final_sha256']
rev=R/'results/presentation-20261014/all-costs-20261015';rev.mkdir(parents=True,exist_ok=True)
for name in ['source.pptx','build.mjs','finalize.mjs','changes.json','validation.json','finalization-result.json','audit.py','audit.json','archive_provider.py','deliver.py']:
 shutil.copy2(B/name,rev/name)
shutil.copytree(B/'final-previews',rev/'final-previews',dirs_exist_ok=True)
out=O/F.name;assert not out.exists();shutil.copy2(F,out)
U=W/'output/PhysLiteBench-all-cost-figures-20261015';dest=O/U.name;assert not dest.exists();shutil.copytree(U,dest)
zp=W/'output'/(U.name+'.zip')
with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(U.rglob('*')):
  if p.is_file():z.write(p,Path(U.name)/p.relative_to(U))
zo=O/zp.name;shutil.copy2(zp,zo);assert sha(F)==sha(out) and sha(zp)==sha(zo)
p=R/'results/presentation-20261014/update-on-completion.json';shutil.copy2(p,rev/'previous-update-on-completion.json');c=json.loads(p.read_text());now=datetime.now(timezone.utc).isoformat()
c.update(current_deck=str(out),current_deck_sha256=sha(out),prior_deck=str(old),prior_deck_sha256=sha(old),updated_utc=now,authoring_script=str(rev/'build.mjs'),authoring_data=str(dest/'resource-summary.json'),authoring_backup=str(rev),final_filename_suggestion=out.name,state='complete')
c['user_request']+=' Latest request: include every Pareto setting, including previously omitted incomplete-cost settings, and plot both dollar and token costs to the best available knowledge with clear uncertainty caveats.'
c['final_update_requirements'][2]='Use reproducible Python dollar and token plots with straight segments and logarithmic cost axes. Draw all settings, including incomplete usage at recorded lower bounds with open markers and finite provider reserve ceilings where available. Frontiers at these coordinates are provisional. Preserve original data and grades. No model runs or grading changes are authorized by this presentation revision.'
c['final_update_requirements'].append('Slides 6 and 7 have dollar and token panels: all 24 benchmark no-hint settings and 36 matched hint settings. Their union is 48 unique settings and 2,068 attempts, not the sum of overlapping views. A one-task, three-attempt DeepSeek None pilot appears separately and is excluded from the primary denominator and frontier. Token totals include input plus output exactly once, with cached input and reasoning already included. For DeepSeek, provider ledgers replace native client usage; do not add the two together. Unknown costs are not zero or imputed complete estimates.')
c['all_cost_revision']={'completed_utc':now,'source_sha256':sha(old),'changed_main_slides':[6,7],'all_no_hint_settings':24,'all_matched_hint_settings':36,'unique_primary_settings':48,'study_attempts_unchanged':2068,'pilot_attempts_displayed_separately':3,'new_trials_added':0,'previously_omitted_no_hint_settings':5,'previously_omitted_hint_settings':14,'plot_metrics':['Recorded API-equivalent USD / attempt','Recorded input plus output tokens / attempt'],'incomplete_usage':'Open markers at recorded minima. Dollar bars terminate at conservative reserves. No imputation. Dashed envelopes use all primary settings at displayed costs and are provisional.','source_files_hashed':26,'provider_batches_archived':157,'provider_unique_requests':3341,'data_bundle':str(dest),'data_bundle_zip':str(zo),'validation':str(rev/'validation.json'),'review':str(rev/'audit.json'),'watcher_unchanged':'PAUSED'}
c['narrative']=c['narrative'].replace('no-hint frontier 6; three-condition frontiers 7','no-hint dollar/token frontiers with all settings 6; three-condition dollar/token frontiers with all settings 7')
c['completion']['presentation_review']=str(rev/'audit.json');c['completion']['presentation_validation']=str(rev/'validation.json');c['completion']['delivery_state']='all_cost_revision_saved_for_delivery_in_this_chat'
p.write_text(json.dumps(c,indent=2)+'\n')
receipt={'presentation':str(out),'sha256':sha(out),'bundle_zip':str(zo),'prior_user_source_preserved':str(old),'revision':str(rev)};(B/'delivery.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
