from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
F=Path('/Users/yp9846/Documents/ChatGPT/phys-lite-bench/.presentation-local-final-20261015')
R=Path('/Users/yp9846/Desktop/PhysLiteBench')
B=R/'results/hint-comparison-20261014'
P=R/'results/presentation-20261014'
source=R/'output/presentation-20261014/PhysLiteBench-10-minute-combined-frontiers.pptx'
final=Path('/Users/yp9846/Documents/ChatGPT/phys-lite-bench/output/physlitebench-presentation/PhysLiteBench-10-minute-collaborator-and-local-hints-final.pptx')
dest=source.parent/final.name
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
review=json.loads((F/'review.json').read_text())
assert review['result']=='passed' and sha(source)==review['source_sha256'] and sha(final)==review['final_sha256']
status=json.loads((B/'status.json').read_text()); assert status['state']=='complete' and status['recorded_attempts']==351 and status['remaining_fresh_attempts']==0
assert json.loads((B/'final-audit-20261015.json').read_text())['result']=='passed'
archive=P/'local-final-20261015';archive.mkdir(exist_ok=True)
contract=P/'update-on-completion.json'
shutil.copy2(contract,archive/'completion-contract-before-final.json')
for n in ['build.mjs','finalize.mjs','inspect.mjs','preserve_workbooks.py','prepare-results.py','plot-final.py','final-audit.py','review.py','release.py','data.json','validation.json','finalization-result.json','workbook-preservation.json','review.json','source.pptx']:
 shutil.copy2(F/n,archive/n)
shutil.copytree(F/'final-previews',archive/'final-previews',dirs_exist_ok=True)
shutil.copy2(final,dest); assert sha(dest)==review['final_sha256']
prep=json.loads((F/'preparation.json').read_text());prep.update(state='complete_validated_and_saved',presentation_edited=True,deck_output_created=True);(F/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n');shutil.copy2(F/'preparation.json',archive/'preparation.json')
c=json.loads(contract.read_text());c.update(prior_deck=str(source),prior_deck_sha256=sha(source),current_deck=str(dest),current_deck_sha256=sha(dest),state='finalized_ready_for_delivery',updated_utc=datetime.now(timezone.utc).isoformat(),authoring_script=str(archive/'build.mjs'),authoring_data=str(archive/'data.json'),authoring_backup=str(archive),timing_current_seconds=c['timing_final_seconds'],finalization_preparation=prep)
c['local_results_slide_strategy']='Completed. Slide 7 contains the separate High-effort comparison with four exact-cost points and all nine success/error counts. Earlier plots and embedded workbooks are preserved.'
c['narrative']=c['narrative'].replace('blank local three-model High-effort comparison on slide 7','completed local three-model High-effort comparison on slide 7')
c['completion']={'completed_observations':351,'local_attempts':313,'imported_historical_controls':38,'benchmark_audit':str(B/'final-audit-20261015.json'),'report':str(B/'FINAL-REPORT.txt'),'figure':str(R/'output/hint-comparison-20261014/hint-comparison-final.png'),'presentation_review':str(archive/'review.json'),'presentation_validation':str(archive/'validation.json'),'delivery_state':'ready','watcher_state':'pending_disable_after_delivery'}
contract.write_text(json.dumps(c,indent=2)+'\n')
artifacts=[dest,B/'FINAL-REPORT.txt',B/'final-summary.csv',B/'final-audit-20261015.json',R/'output/hint-comparison-20261014/hint-comparison-final.png',R/'output/hint-comparison-20261014/hint-comparison-final.pdf',R/'output/hint-comparison-20261014/hint-comparison-final.svg']
release={'created_utc':datetime.now(timezone.utc).isoformat(),'state':'ready_for_delivery','benchmark_status':status,'artifacts':[{'path':str(p),'sha256':sha(p)} for p in artifacts],'presentation_review':str(archive/'review.json'),'prior_deck_preserved':str(source),'no_push':True}
(B/'final-delivery-20261015.json').write_text(json.dumps(release,indent=2)+'\n')
print(json.dumps({'final_presentation':str(dest),'sha256':sha(dest),'archive':str(archive),'status':'ready_for_delivery'},indent=2))
