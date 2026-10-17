from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, shutil, zipfile
W=Path('/Users/yp9846/Documents/ChatGPT/phys-lite-bench')
B=W/'.presentation-no-pilot-20261016'
R=Path('/Users/yp9846/Desktop/PhysLiteBench')
O=R/'output/presentation-20261014'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=O/'PhysLiteBench-10-minute-all-costs-reviewed.pptx'
assert sha(source)==sha(B/'source.pptx'), 'User saved further edits; re-import before delivery'
F=W/'output/physlitebench-presentation/PhysLiteBench-10-minute-no-pilot.pptx'
A=json.loads((B/'audit.json').read_text())
assert sha(F)==A['final_sha256']
U=W/'output/PhysLiteBench-no-pilot-figures-20261016'
dest=O/U.name
out=O/F.name
assert not dest.exists() and not out.exists()
shutil.copy2(F,out)
shutil.copytree(U,dest)
rev=R/'results/presentation-20261014/no-pilot-20261016'
rev.mkdir(parents=True,exist_ok=True)
for name in ['source.pptx','build.mjs','finalize.mjs','changes.json','validation.json','finalization-result.json','audit.json','deliver.py']:
    shutil.copy2(B/name,rev/name)
shutil.copytree(B/'final-previews',rev/'final-previews',dirs_exist_ok=True)
with zipfile.ZipFile(O/(U.name+'.zip'),'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(U.rglob('*')):
        if p.is_file():z.write(p,Path(U.name)/p.relative_to(U))
p=R/'results/presentation-20261014/update-on-completion.json'
shutil.copy2(p,rev/'previous-update-on-completion.json')
c=json.loads(p.read_text())
now=datetime.now(timezone.utc).isoformat()
c.update(current_deck=str(out),current_deck_sha256=sha(out),prior_deck=str(source),prior_deck_sha256=sha(source),updated_utc=now,authoring_script=str(rev/'build.mjs'),authoring_data=str(dest/'resource-summary.json'),authoring_backup=str(rev),final_filename_suggestion=out.name)
c['user_request']+=' Latest request: remove the one-task pilot crosses and labels from both Python panels on slide 6. Preserve all primary data and latest user text edits.'
c['final_update_requirements']=[v.replace('A one-task, three-attempt DeepSeek None pilot appears separately and is excluded from the primary denominator and frontier.','The one-task, three-attempt DeepSeek None pilot is omitted from the plots and excluded from the primary denominator and frontier. Its source observations are preserved.') for v in c['final_update_requirements']]
c['pilot_display_revision']={'completed_utc':now,**A,'data_bundle':str(dest),'figure':str(dest/'figures/09-no-hint-all-costs.png'),'validation':str(rev/'validation.json')}
c['user_edits_observed']={'observed_utc':now,'current_sha256':sha(source),'preserved_snapshot':str(rev/'source.pptx'),'note':'Imported latest saved deck, including updated bead/covariance wording. All 18 other slide renders unchanged.'}
c['completion'].update(presentation_review=str(rev/'audit.json'),presentation_validation=str(rev/'validation.json'),delivery_state='no_pilot_revision_saved_for_delivery_in_this_chat')
p.write_text(json.dumps(c,indent=2)+'\n')
assert sha(out)==sha(F)
receipt={'presentation':str(out),'figure':str(dest/'figures/09-no-hint-all-costs.png'),'bundle':str(dest)}
(B/'delivery.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
