from pathlib import Path
import collections,hashlib,json,re,zipfile,xml.etree.ElementTree as E
W=Path('/Users/yp9846/Documents/ChatGPT/phys-lite-bench');R=Path('/Users/yp9846/Desktop/PhysLiteBench');B=W/'.presentation-plain-20261015'
source=R/'output/presentation-20261014/PhysLiteBench-10-minute-CS-audience.pptx';final=W/'output/physlitebench-presentation/PhysLiteBench-10-minute-plain-language-v3.pptx'
A='http://schemas.openxmlformats.org/drawingml/2006/main';ns={'a':A};fonts=collections.Counter();timing=[];titles=[];notes=[]
def root(z,p):return E.fromstring(z.read(p))
def tx(r):return [x.text or '' for x in r.findall('.//a:t',ns)]
def table(r):return [[tx(c) for c in row.findall('a:tc',ns)] for row in r.findall('.//a:tbl/a:tr',ns)]
with zipfile.ZipFile(source) as src,zipfile.ZipFile(final) as z:
 assert tx(root(z,'ppt/slides/slide1.xml'))==['PhysLiteBench','Yue Pan and Andrew Liu']
 for old,new in [(1,2),(4,5),(5,6),(12,13),(13,14)]:assert table(root(src,f'ppt/slides/slide{old}.xml'))==table(root(z,f'ppt/slides/slide{new}.xml'))
 records=json.loads((B/'workbook-preservation.json').read_text())['records'];assert len(records)==4
 for r in records:assert hashlib.sha256(z.read(r['workbook'])).hexdigest()==r['sha256']
 for i in range(1,17):
  slide=tx(root(z,f'ppt/slides/slide{i}.xml'));titles.append(slide[0]);nt=' '.join(tx(root(z,f'ppt/notesSlides/notesSlide{i}.xml')));notes.append(nt)
  m=re.search(r'Target time: (\d+) seconds',nt);timing.append(int(m.group(1)) if m else 0)
  if i in [3,4,10]:
   maintext=' '.join(slide)+' '+nt.split('Sources:')[0]
   for forbidden in ['stationary increments','quenched','covariance','deterministic preparation','Poisson']:
    assert forbidden.lower() not in maintext.lower(),(i,forbidden)
 assert sum(timing[:11])==600 and timing[11]==180
 assert 'Pending completion' in tx(root(z,'ppt/slides/slide7.xml'))
 assert titles[15].startswith('Backup:')
 for p in z.namelist():
  if p.startswith('ppt/') and p.endswith('.xml'):
   for x in root(z,p).iter():
    f=x.get('typeface')
    if f and not f.startswith('+'):fonts[f]+=1
 assert set(fonts)=={'Times New Roman'}
diag=next(x for x in json.loads((R/'results/trajectory-analysis-20261014/diagnostic-repairs.json').read_text())['cases'] if x['task']=='single-file-memory')
assert round(100*max(diag['before'].values()))==136
assert f"{100*max(diag['after'].values()):.3f}"=='0.003'
audit={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'final':str(final),'final_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),'title_and_authors_only':True,'authors':['Yue Pan','Andrew Liu'],'total_slides':16,'main_content_slides':10,'title_slide':1,'example_slides':[3,4,10],'hint_results_slide':7,'hint_results_pending':True,'questions_slide':12,'backup_slides':[13,14,15,16],'analytical_example_chart_moved_to_backup':16,'main_explanation_has_no_specialist_terms':True,'times_new_roman_only':dict(fonts),'all_four_source_workbooks_byte_identical':True,'rounded_diagnostic_errors_verified':True,'main_seconds_including_title':600,'questions_seconds':180,'timing_seconds':timing,'titles':titles,'visual_review':{'all_16_final_slides_inspected':True,'layout':'No unintended clipping or overlap. Title has only title and authors. Example uses three snapshots, and failure flowchart highlights the retained formula.','native_powerpoint_tested':False,'remaining_validator_warnings':'Conservative backup-table overflow estimates on slides 13 and 14. Rendered tables fit above footnotes.'},'scientific_scope':'Beads are a schematic for the documented point-particle model. Main story describes spread at individual times versus the relation between two times. Precise long-time definition and equations are retained in backup notes. Analyst correction is not a model pass.'}
(B/'content-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({k:audit[k] for k in ['final_sha256','title_and_authors_only','main_seconds_including_title','main_explanation_has_no_specialist_terms','all_four_source_workbooks_byte_identical']},indent=2))
