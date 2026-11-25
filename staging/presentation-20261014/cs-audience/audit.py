from pathlib import Path
import collections,hashlib,json,math,re,zipfile,xml.etree.ElementTree as E
W=Path('/Users/yp9846/Documents/ChatGPT/phys-lite-bench');R=Path('/Users/yp9846/Desktop/PhysLiteBench');B=W/'.presentation-cs-20261015'
source=R/'output/presentation-20261014/PhysLiteBench-10-minute-physics-example.pptx'
final=W/'output/physlitebench-presentation/PhysLiteBench-10-minute-CS-audience-v2.pptx'
A='http://schemas.openxmlformats.org/drawingml/2006/main';C='http://schemas.openxmlformats.org/drawingml/2006/chart';ns={'a':A,'c':C}
def root(z,p):return E.fromstring(z.read(p))
def texts(r):return [x.text or '' for x in r.findall('.//a:t',ns)]
def table(r):return [[texts(c) for c in row.findall('a:tc',ns)] for row in r.findall('.//a:tbl/a:tr',ns)]
fonts=collections.Counter();titles=[];timing=[];checks=[]
with zipfile.ZipFile(source) as src,zipfile.ZipFile(final) as z:
 for old,new in [(6,5),(14,12),(15,13)]:
  assert table(root(src,f'ppt/slides/slide{old}.xml'))==table(root(z,f'ppt/slides/slide{new}.xml'))
  checks.append({'original_slide':old,'revised_slide':new,'table_contents_preserved':True})
 preserved=json.loads((B/'workbook-preservation.json').read_text())['records'];assert len(preserved)==3
 for p in preserved:assert hashlib.sha256(z.read(p['workbook'])).hexdigest()==p['sha256']
 for i in range(1,15):
  titles.append(texts(root(z,f'ppt/slides/slide{i}.xml'))[0])
  note=' '.join(texts(root(z,f'ppt/notesSlides/notesSlide{i}.xml')))
  m=re.search(r'Target time: (\d+) seconds',note);timing.append(int(m.group(1)) if m else 0)
 assert sum(timing[:10])==600 and timing[10]==180
 assert titles[1]=='Example: particles that cannot pass' and titles[5]=='Pareto frontiers with hints' and titles[8]=='The example again: the wrong memory rule'
 assert 'Pending completion' in texts(root(z,'ppt/slides/slide6.xml'))
 assert not root(z,'ppt/slides/slide6.xml').findall('.//c:chart',ns)
 for p in z.namelist():
  if p.startswith('ppt/') and p.endswith('.xml'):
   for x in root(z,p).iter():
    f=x.get('typeface')
    if f and not f.startswith('+'):fonts[f]+=1
 assert set(fonts)=={'Times New Roman'}
 plot=root(z,'ppt/slides/charts/chart1.xml')
 for idx,ser in enumerate(plot.findall('.//c:ser',ns)):
  xv=[float(v.text) for v in ser.findall('.//c:xVal//c:pt/c:v',ns)]
  yv=[float(v.text) for v in ser.findall('.//c:yVal//c:pt/c:v',ns)]
  expected=[(1+math.sqrt(r)-math.sqrt(r-1))/2 if idx==0 else (math.sqrt(1+r)-math.sqrt(r-1))/math.sqrt(2) for r in xv]
  assert len(yv)==57 and max(abs(a-b) for a,b in zip(yv,expected))<5.1e-10
  assert yv[0]==1
traj=json.loads((R/'jobs/pareto-gpt-5.6-terra-single-file-memory-20261011-135517/single-file-memory__PC6LumP/agent/trajectory.json').read_text())
quote='The supplied covariance expression matches the deterministic (quenched) single-file limit.'
assert quote in next(s['message'] for s in traj['steps'] if s['step_id']==8)
assert '1 passed' in str(next(s for s in traj['steps'] if s['step_id']==9)['observation'])
metrics=json.loads((R/'results/trajectory-analysis-20261014/cases/single-file-memory__PC6LumP/metrics.json').read_text())
diag=next(c for c in json.loads((R/'results/trajectory-analysis-20261014/diagnostic-repairs.json').read_text())['cases'] if c['task']=='single-file-memory')
assert f"{100*max(diag['before'].values()):.2f}%"=='136.06%'
assert f"{100*max(diag['after'].values()):.4f}%"=='0.0028%'
assert f"{100*metrics['parameter_relative_error']:.4f}%"=='0.0055%'
plan=json.loads((R/'results/hint-comparison-20261014/plan.json').read_text())
for f in ['environment/README.md','environment/model.py','hint.md','solution/model.py']:
 assert hashlib.sha256((R/'tasks/single-file-memory'/f).read_bytes()).hexdigest()==plan['tasks']['single-file-memory']['source_sha256'][f]
audit={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'final':str(final),'final_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),'main_slides':10,'total_slides':14,'titles':titles,'example_slides':[2,3,9],'hint_results_slide':6,'hint_results_pending':True,'all_visible_fonts':dict(fonts),'original_workbooks_preserved_exactly':3,'new_analytic_charts':1,'analytic_covariance_curves_verified':True,'tables':checks,'recorded_agent_statement_and_public_test_verified':True,'diagnostic_values_verified':True,'example_matches_frozen_hint_study_files':True,'main_seconds':600,'questions_seconds':180,'timing_seconds':timing,'visual_review':{'all_14_final_slides_inspected':True,'layout':'No unintended clipping or overlap in reimported full-size previews','native_powerpoint_tested':False,'remaining_validator_warnings':'Conservative native-table overflow estimates on unchanged backup tables 12 and 13. Rendered tables fit above footnotes.'},'scientific_scope':'Saved actions and final source only. Analyst correction is not a model pass or a hint result. Example curve is the normalized long-time covariance coefficient at admissible time multipliers, not finite-time trajectory data.'}
(B/'content-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({k:audit[k] for k in ['final_sha256','main_slides','total_slides','main_seconds','hint_results_pending','analytic_covariance_curves_verified','original_workbooks_preserved_exactly']},indent=2))
