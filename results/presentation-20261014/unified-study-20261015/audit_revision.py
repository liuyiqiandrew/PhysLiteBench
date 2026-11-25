from pathlib import Path
from xml.etree import ElementTree as ET
import hashlib,json,re,zipfile,posixpath
W=Path('/Users/yp9846/Documents/ChatGPT/phys-lite-bench');B=W/'.presentation-unified-20261015';U=W/'output/PhysLiteBench-study-figures-20261015';R=Path('/Users/yp9846/Desktop/PhysLiteBench')
source=B/'source.pptx';live=R/'output/presentation-20261014/PhysLiteBench-10-minute-collaborator-and-local-hints-final.pptx';final=W/'output/physlitebench-presentation/PhysLiteBench-10-minute-unified-study.pptx'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(source)==sha(live),'New user save needs review'
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
def texts(z,n):return [e.text or '' for e in ET.fromstring(z.read(f'ppt/slides/slide{n}.xml')).findall('.//a:t',ns)]
old=zipfile.ZipFile(source);new=zipfile.ZipFile(final)
unchanged=[1,3,12,15]
for n in unchanged:assert texts(old,n)==texts(new,n),f'Unintended text change on slide {n}'
wbdir=B/'original-chart-workbooks';wbdir.mkdir(exist_ok=True);wbs=[]
for n in old.namelist():
 if n.startswith('ppt/embeddings/') and n.endswith('.xlsx'):
  p=wbdir/Path(n).name;p.write_bytes(old.read(n));wbs.append({'source_package_entry':n,'file':p.name,'sha256':sha(p)})
assert len(wbs)==7
(B/'original-chart-workbooks/manifest.json').write_text(json.dumps(wbs,indent=2)+'\n')
figs={6:'01-no-hint-frontier',7:'02-hint-frontiers',8:'03-luna-effort-hints',9:'04-outcomes-by-model',16:'05-bead-covariance',17:'06-no-hint-efforts'}
for n,name in figs.items():
 sl=ET.fromstring(new.read(f'ppt/slides/slide{n}.xml'))
 assert not sl.findall('.//a:graphicData/*[@{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id]',ns)
 rels={r.attrib['Id']:r.attrib['Target'] for r in ET.fromstring(new.read(f'ppt/slides/_rels/slide{n}.xml.rels'))}
 pics=sl.findall('.//p:pic//a:blip',ns);want=sha(U/f'figures/{name}.png')
 hashes=[hashlib.sha256(new.read(posixpath.normpath(posixpath.join('ppt/slides',rels[b.attrib['{'+ns['r']+'}embed']])).lstrip('/'))).hexdigest() for b in pics]
 assert want in hashes,('Chart bytes differ',n)
for n in range(1,18):
 t=' '.join(texts(new,n));assert not re.search(r'\b(local|collaborator|pending)\b',t,re.I),(n,t)
assert 'Implemented with the Harbor framework.' in texts(new,2)
assert any('covariance' in t for t in texts(new,10))
D=json.loads((U/'summary.json').read_text());m=D['metadata'];rows=D['summary']
assert sum(r['attempts'] for r in rows)==2068
assert [sum(r[k] for r in rows) for k in ['passes','graded_failures','errors']]==[899,904,265]
assert len(rows)==48 and len({r['model'] for r in rows})==7
assert len(json.loads((U/'deduplication-audit.json').read_text())['removed'])==38
assert len([r for r in rows if r['cost_complete']])==32
for r in rows:
 assert r['passes']+r['graded_failures']+r['errors']==r['attempts']
 assert (r['mean_cost_usd'] is not None)==r['cost_complete']
 assert r['tasks']==13
 assert r['mean_cost_lower_usd']>=0
 if r['mean_cost_upper_usd'] is not None:assert r['mean_cost_upper_usd']+1e-9>=r['mean_cost_lower_usd']
for f in m['all_no_hint_frontier']:
 assert not any(q['cost_complete'] and q['condition']=='baseline' and q['mean_cost_usd']<=f['mean_cost_usd'] and q['pass_rate']>=f['pass_rate'] and (q['mean_cost_usd']<f['mean_cost_usd'] or q['pass_rate']>f['pass_rate']) for q in rows)
manifest=json.loads((U/'source-manifest.json').read_text())
for e in manifest:assert sha(U/e['file'])==e['sha256']
patterns=[re.compile(rb'(?<![A-Za-z0-9])pit_[A-Za-z0-9_-]{20,}'),re.compile(rb'(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}'),re.compile(rb'Bearer\s+[A-Za-z0-9._-]{20,}'),re.compile(rb'-----BEGIN [A-Z ]*PRIVATE KEY-----')]
scanned=0
for p in U.rglob('*'):
 if p.is_file() and p.suffix.lower() in ['.json','.csv','.py','.txt','.md','.svg']:
  scanned+=1;assert not any(q.search(p.read_bytes()) for q in patterns),'Credential-like text in '+str(p)
for n in new.namelist():
 if n.endswith(('.xml','.rels')):assert not any(q.search(new.read(n)) for q in patterns),'Credential-like text in '+n
notes=[]
for n in range(1,12):
 t=' '.join(ET.fromstring(new.read(f'ppt/notesSlides/notesSlide{n}.xml')).itertext());hit=re.search(r'Target time: (\d+) seconds',t);assert hit,n;notes.append(int(hit[1]))
assert sum(notes)==600
report={'source_sha256':sha(source),'final_sha256':sha(final),'live_user_source_unchanged_at_audit':True,'unchanged_visible_text_slides':unchanged,'all_17_final_slides_visually_reviewed':True,'chart_image_bytes_match_python_outputs':list(figs),'source_files_hashed':len(manifest),'original_workbooks_archived':len(wbs),'attempts':2068,'settings':48,'model_versions':7,'known_duplicate_trials_removed':38,'outcomes':{'passes':899,'graded_failures':904,'execution_errors':265},'timing_seconds':notes,'main_talk_seconds':600,'qa_seconds':180,'credential_pattern_scan':{'files':scanned,'findings':0},'native_application_open_tested':False,'user_requested_Python_figures':True,'charts_editable_via':'Archived Python scripts and data; slide charts are rendered images.','notes':'Visible slide language groups by model, effort, condition. Runtime and provenance caveats remain in notes and source bundle. Source chart workbooks retained. No benchmark runs or grades changed.'}
(B/'revision-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
