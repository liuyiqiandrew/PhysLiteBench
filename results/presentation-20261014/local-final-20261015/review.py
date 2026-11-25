from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,re,zipfile,xml.etree.ElementTree as ET
F=Path('/Users/yp9846/Documents/ChatGPT/phys-lite-bench/.presentation-local-final-20261015')
R=Path('/Users/yp9846/Desktop/PhysLiteBench')
source=R/'output/presentation-20261014/PhysLiteBench-10-minute-combined-frontiers.pptx'
final=Path('/Users/yp9846/Documents/ChatGPT/phys-lite-bench/output/physlitebench-presentation/PhysLiteBench-10-minute-collaborator-and-local-hints-final.pptx')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(source)=='0b5ffc3feeec95dbe2b6fd041f9d863b92359402ab8ffc664fc68529c51933df'
assert sha(final)=='becc4df0ff66089fe4565c2fb9fcaa87b64ecfb7af1444c0461ef7cc56aec551'
audit=json.loads((F/'final-audit-20261015.json').read_text()); assert audit['result']=='passed'
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
unchanged=[1,2,3,4,6,8,9,10,12,13,14,15,16,17]
with zipfile.ZipFile(source) as a,zipfile.ZipFile(final) as b:
 def text(z,p): return [x.text or '' for x in ET.fromstring(z.read(p)).findall('.//a:t',ns)]
 for n in unchanged:
  assert text(a,f'ppt/slides/slide{n}.xml')==text(b,f'ppt/slides/slide{n}.xml'),('changed slide',n)
 sourcebooks=Counter(hashlib.sha256(a.read(n)).hexdigest() for n in a.namelist() if n.endswith('.xlsx'))
 finalbooks=Counter(hashlib.sha256(b.read(n)).hexdigest() for n in b.namelist() if n.endswith('.xlsx'))
 assert not sourcebooks-finalbooks
 def vals(z,n):
  x=ET.fromstring(z.read(f'ppt/slides/charts/chart{n}.xml'))
  return [(e.tag.rsplit('}',1)[-1],e.text) for e in x.iter() if e.tag.rsplit('}',1)[-1] in ['f','v','t']]
 for p,q in [(1,1),(2,2),(3,4),(4,5),(5,6),(6,7)]: assert vals(a,p)==vals(b,q),(p,q)
 timings=[]
 for n in range(1,13):
  note=' '.join(text(b,f'ppt/notesSlides/notesSlide{n}.xml'))
  timings.append(int(re.search(r'Target time: (\d+) seconds',note).group(1)))
 assert timings==[10,45,85,60,55,75,50,55,55,80,30,180]
 allxml=b'\n'.join(b.read(n) for n in b.namelist() if n.endswith(('.xml','.rels')))
 hits=[]
 for pat in [rb'pit_[A-Za-z0-9_-]{20,}',rb'sk-[A-Za-z0-9_-]{20,}',rb'Bearer\s+[A-Za-z0-9_-]{20,}']:
  hits+=re.findall(pat,allxml)
 assert not hits
r={'reviewed_utc':datetime.now(timezone.utc).isoformat(),'result':'passed','source_sha256':sha(source),'final_sha256':sha(final),'visible_text_preserved_slides':unchanged,'user_edits_preserved_slides':[3,10,16],'six_original_chart_data_series_preserved':True,'original_workbooks_preserved_exactly':sum(sourcebooks.values()),'final_workbook_count':sum(finalbooks.values()),'main_talk_seconds':sum(timings[:11]),'questions_seconds':timings[11],'all_17_final_slides_visually_inspected':True,'native_powerpoint_open_verified':False,'layout_warning_review':{'slides':[13,14],'resolution':'Existing backup tables fit in the final rendered slides with visible footers; preserved the user-approved layout.'},'credential_pattern_matches':0,'benchmark_audit_result':audit['result'],'remaining_attempts':0}
(F/'review.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
