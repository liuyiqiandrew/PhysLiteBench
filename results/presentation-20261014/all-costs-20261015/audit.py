from pathlib import Path
import hashlib,json,re,zipfile,xml.etree.ElementTree as E
W=Path('/Users/yp9846/Documents/ChatGPT/phys-lite-bench');B=W/'.presentation-costs-20261015';U=W/'output/PhysLiteBench-all-cost-figures-20261015';F=W/'output/physlitebench-presentation/PhysLiteBench-10-minute-all-costs-reviewed.pptx'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=Path('/Users/yp9846/Desktop/PhysLiteBench/output/presentation-20261014/PhysLiteBench-10-minute-physics-failures.pptx');assert sha(source)==sha(B/'source.pptx')=='91fdd30a8254881b87ecbe98d786291f294d5099d9f8388ae921a266bb9b4687'
unchanged=[n for n in range(1,20) if n not in [6,7]]
for n in unchanged:assert sha(B/f'before/slide-{n:02d}.png')==sha(B/f'final-previews/slide-{n:02d}.png'),n
manifest=json.loads((U/'source-manifest.json').read_text())
for m in manifest:assert sha(U/m['file'])==m['sha256']
D=json.loads((U/'resource-summary.json').read_text());old=json.loads((U/'summary.json').read_text());A=json.loads((U/'resource-plot-audit.json').read_text());rows=D['summary'];keys=lambda rs:{(r['model'],r['effort'],r['condition']) for r in rs};assert len(rows)==48 and keys(rows)==keys(old['summary']) and sum(r['attempts'] for r in rows)==2068
for r,q in zip(rows,old['summary']):
 for k,v in q.items():assert r[k]==v
 assert r['plot_usd']==(r['mean_cost_usd'] if r['cost_complete'] else r['mean_cost_lower_usd'])
 assert r['plot_tokens']==r['recorded_total_tokens']/r['attempts']
 if r['recorded_input_tokens'] is not None:assert r['recorded_total_tokens']==r['recorded_input_tokens']+r['recorded_output_tokens']
 assert r['plot_usd']>0 and r['plot_tokens']>0
for metric in ['usd','tokens']:
 panels=[p for p in A['panels'] if p['metric']==metric and p['compact']];assert len(panels)==2
 sets=[{tuple(k) for k in p['plotted_primary_settings']} for p in panels];assert sorted(map(len,sets))==[24,36] and len(sets[0]|sets[1])==48 and len(sets[0]&sets[1])==12
 assert all(p['pilot_shown']==(p['scope']=='base') for p in panels)
assert D['pilot']['attempts']==3 and D['pilot']['tasks']==1
new_deepseek=sum(c['recorded_tokens'] for c in D['components'] if c['provider_request_count']);old_deepseek=sum(c['native_recorded_tokens_before_provider_reconciliation'] for c in D['components'] if c['provider_request_count']);assert new_deepseek>=old_deepseek
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
with zipfile.ZipFile(F) as z:
 slides=[n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',n)];assert len(slides)==19
 images={hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('ppt/media/')}
 for name in ['03-luna-effort-hints','07-reviewed-outcomes','08-physics-failure-matrix','09-no-hint-all-costs','10-hint-all-costs']:assert sha(U/'figures'/f'{name}.png') in images
 notes={n:''.join(E.fromstring(z.read(f'ppt/notesSlides/notesSlide{n}.xml')).itertext()) for n in range(1,20)}
 times=[int(re.search(r'Target time: (\d+) seconds',notes[n]).group(1)) for n in range(1,12)];assert sum(times)==600 and 'Target time: 180 seconds' in notes[12]
 scans=[(n,z.read(n).decode('utf8')) for n in z.namelist() if n.endswith('.xml')]
patterns=[re.compile(r'\bsk-[A-Za-z0-9_-]{20,}'),re.compile(r'\bpit_[A-Za-z0-9]{20,}'),re.compile(r'\bBearer\s+[A-Za-z0-9._-]{20,}')]
for p in U.rglob('*'):
 if p.is_file() and p.suffix.lower() in {'.py','.md','.json','.csv','.txt','.svg'}:scans.append((str(p.relative_to(U)),p.read_text()))
findings=[name for name,t in scans if any(p.search(t) for p in patterns)];assert not findings,findings
V=json.loads((B/'validation.json').read_text());assert V['packageIntegrity']['status']=='pass' and V['presentationLayout']['finding_count']==0
report={'source_sha256':sha(source),'final_sha256':sha(F),'source_user_changes_preserved':True,'unchanged_visible_slides_pixel_identical':unchanged,'changed_visible_slides':[6,7],'slide_count':19,'main_talk_seconds':600,'qa_seconds':180,'source_hashes_verified':len(manifest),'new_figure_bytes_match':True,'no_hint_settings_in_each_metric':24,'matched_hint_settings_in_each_metric':36,'unique_primary_settings':48,'primary_attempts':2068,'pilot_attempts_displayed_separately':3,'new_trials_added':0,'no_hint_partial_cost_points':5,'hint_partial_cost_points':14,'additional_provider_tokens_recovered':new_deepseek-old_deepseek,'provider_and_client_usage_not_added_together':True,'cached_and_reasoning_tokens_not_added_twice':True,'frontiers':'Provisional envelopes at displayed recorded costs, including incomplete lower bounds. Single-task pilot excluded.','credential_scan_findings':0,'powerpoint_native_open_tested':False,'visual_review':{'slides':list(range(1,20)),'all_final_slides_reviewed_individually':True,'findings':'Figures use labelled dollar and token panels with uncertainty shown. No clipping or unintended changes found. Unchanged tables retain pre-existing heuristic size warnings and render identically to the source.'}}
(B/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['slide_count','no_hint_settings_in_each_metric','matched_hint_settings_in_each_metric','unique_primary_settings','primary_attempts','pilot_attempts_displayed_separately','additional_provider_tokens_recovered','credential_scan_findings']},indent=2))
