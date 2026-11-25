from pathlib import Path
import hashlib,json,re,zipfile,xml.etree.ElementTree as E
W=Path('/Users/yp9846/Documents/ChatGPT/phys-lite-bench');B=W/'.presentation-failures-20261015';U=W/'output/PhysLiteBench-physics-failure-figures-20261015';F=W/'output/physlitebench-presentation/PhysLiteBench-10-minute-physics-failures.pptx'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=Path('/Users/yp9846/Desktop/PhysLiteBench/output/presentation-20261014/PhysLiteBench-10-minute-focused.pptx')
assert sha(source)==sha(B/'source.pptx')=='a44091641b17d9aa668901fdad53c01e4ff2c3282edf25b815ac35918b4a296e'
unchanged=[1,2,3,4,5,6,7,8,10,12,13,14,15,16,17]
for n in unchanged:assert sha(B/f'before/slide-{n:02d}.png')==sha(B/f'final-previews/slide-{n:02d}.png')
manifest=json.loads((U/'source-manifest.json').read_text())
for m in manifest:assert sha(U/m['file'])==m['sha256']
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
with zipfile.ZipFile(F) as z:
 slides=[n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',n)];assert len(slides)==19
 root=E.fromstring(z.read('ppt/slides/slide19.xml'));rows=root.findall('.//a:tbl/a:tr',ns)
 table=[[''.join(t.itertext()) for t in []]] if False else [[''.join(c.find('.//a:txBody',ns).itertext()) for c in row.findall('a:tc',ns)] for row in rows]
 nums=[[int(v.replace(',','')) for v in row[1:]] for row in table[1:]]
 assert nums==[[84,296,8,1,1,390],[183,158,34,14,1,390],[257,66,57,10,0,390],[524,520,99,25,2,1170]]
 assert all(sum(r[:-1])==r[-1] for r in nums)
 assert [sum(r[c] for r in nums[:3]) for c in range(6)]==nums[3]
 images={hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('ppt/media/')}
 for name in ['03-luna-effort-hints','07-reviewed-outcomes','08-physics-failure-matrix']:assert sha(U/'figures'/f'{name}.png') in images
 notes={n:''.join(E.fromstring(z.read(f'ppt/notesSlides/notesSlide{n}.xml')).itertext()) for n in range(1,20)}
 main_times=[int(re.search(r'Target time: (\d+) seconds',notes[n]).group(1)) for n in range(1,12)];assert sum(main_times)==600
 assert 'Target time: 180 seconds' in notes[12]
 text9=''.join(E.fromstring(z.read('ppt/slides/slide9.xml')).itertext())
 assert all(s not in text9 for s in ['186 / 191','229 / 315'])
 scans=[(n,z.read(n).decode('utf8')) for n in z.namelist() if n.endswith('.xml')]
patterns=[re.compile(r'\bsk-[A-Za-z0-9_-]{20,}'),re.compile(r'\bpit_[A-Za-z0-9]{20,}'),re.compile(r'\bBearer\s+[A-Za-z0-9._-]{20,}')]
for p in U.rglob('*'):
 if p.is_file() and p.suffix.lower() in {'.py','.md','.json','.csv','.txt','.svg'}:scans.append((str(p.relative_to(U)),p.read_text()))
findings=[name for name,t in scans if any(p.search(t) for p in patterns)];assert not findings,findings
D=json.loads((U/'failure-breakdown.json').read_text());A=D['overlap_audit'];assert A['new_trials_added']==0 and A['full_study_attempts_unchanged']==2068 and A['baseline_trials_matched_by_identity']==390 and A['hinted_attempts_reconciled_to_unchanged_published_summary']==780
assert len(D['rows'])==30 and all(r['attempts']==39 for r in D['rows'])
V=json.loads((B/'validation.json').read_text());assert V['packageIntegrity']['status']=='pass' and V['presentationLayout']['finding_count']==0
report={'source_sha256':sha(source),'final_sha256':sha(F),'source_user_changes_preserved':True,'unchanged_visible_slides_pixel_identical':unchanged,'slide_count':19,'main_talk_seconds':600,'qa_seconds':180,'backup_slides':[13,14,15,16,17,18,19],'source_hashes_verified':len(manifest),'native_slide19_table_arithmetic_verified':True,'new_figure_bytes_match':True,'slide8_preserved':True,'slide8_attempts_per_cell':39,'slide8_high_passes':[4,18,30],'classified_existing_attempts':1170,'full_study_attempts':2068,'new_trials_added':0,'baseline_identity_matches':390,'hinted_aggregate_matches':780,'published_trajectory_classifications_independently_reaudited':False,'credential_scan_findings':0,'powerpoint_native_open_tested':False,'visual_review':{'slides':list(range(1,20)),'all_final_slides_reviewed_individually':True,'findings':'No clipping or unintended visual changes found. Updated slides 9 and 11 plus backups 18 and 19. The pre-existing table-size heuristic warnings on slides 5, 13 and 14 do not appear in the rendered slides, whose previews are byte-identical to the source.'},'classification_limit':'Strict primary physics-error taxonomy applies only to the 1170 published reviewed matrix attempts. No imputation for other attempts. Historical controls and runtime differences remain.'}
(B/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['slide_count','full_study_attempts','new_trials_added','baseline_identity_matches','hinted_aggregate_matches','credential_scan_findings','source_hashes_verified']},indent=2))
