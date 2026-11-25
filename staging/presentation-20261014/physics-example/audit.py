from pathlib import Path
import collections,hashlib,json,math,re,zipfile,xml.etree.ElementTree as E
W=Path('/Users/yp9846/Documents/ChatGPT/phys-lite-bench');R=Path('/Users/yp9846/Desktop/PhysLiteBench');B=W/'.presentation-example-20261015'
source=R/'output/presentation-20261014/PhysLiteBench-10-minute-committee-style.pptx'
final=W/'output/physlitebench-presentation/PhysLiteBench-10-minute-physics-example.pptx'
A='http://schemas.openxmlformats.org/drawingml/2006/main';C='http://schemas.openxmlformats.org/drawingml/2006/chart';ns={'a':A,'c':C}
def root(z,p):return E.fromstring(z.read(p))
def texts(r):return [x.text or '' for x in r.findall('.//a:t',ns)]
def table(r):return [[texts(c) for c in row.findall('a:tc',ns)] for row in r.findall('.//a:tbl/a:tr',ns)]
fontcounts=collections.Counter();notes=[];timings=[];checks=[]
with zipfile.ZipFile(source) as src,zipfile.ZipFile(final) as z:
 for old,new in [(2,2),(3,5),(4,6),(9,10),(13,14),(14,15)]:
  assert table(root(src,f'ppt/slides/slide{old}.xml'))==table(root(z,f'ppt/slides/slide{new}.xml'))
  checks.append({'original_slide':old,'revised_slide':new,'native_table_content_identical':True})
 for p in src.namelist():
  if p.startswith('ppt/embeddings/'):assert src.read(p)==z.read(p)
 titles=[]
 for i in range(1,17):
  r=root(z,f'ppt/slides/slide{i}.xml');titles.append(texts(r)[0]);nt=' '.join(texts(root(z,f'ppt/notesSlides/notesSlide{i}.xml')))
  notes.append(nt);m=re.search(r'Target time: (\d+) seconds',nt);timings.append(int(m.group(1)) if m else 0)
 assert sum(timings[:12])==600 and timings[12]==180,timings
 assert not any('Targeted corrections identify' in t for t in titles)
 assert texts(root(z,'ppt/slides/slide11.xml'))==['Hint comparison results','11','Pending completion']
 assert [titles[i-1] for i in [3,4,6,7,8,9,10]]==['Entropy anomaly: the apparatus','Entropy anomaly: the agent’s task','Observed cost–success frontier','More reasoning does not guarantee more passes','The failure modes differ sharply by model','Entropy anomaly: the observed failure','The hint comparison holds effort at High']
 for p in z.namelist():
  if p.startswith('ppt/') and p.endswith('.xml'):
   for x in root(z,p).iter():
    f=x.get('typeface')
    if f and not f.startswith('+'):fontcounts[f]+=1
 assert set(fontcounts)=={'Times New Roman'},fontcounts
 for i in [1,2]:
  cr=root(z,f'ppt/slides/charts/chart{i}.xml')
  for idx,ser in enumerate(cr.findall('.//c:ser',ns)):
   xv=[float(v.text) for v in ser.findall('.//c:xVal//c:pt/c:v',ns)]
   yv=[float(v.text) for v in ser.findall('.//c:yVal//c:pt/c:v',ns)]
   expected=[(1+.4*(idx-1))*(1+.4*math.cos(2*math.pi*x)) if i==1 else .5*(1+math.cos(2*math.pi*x)) for x in xv]
   assert len(yv)==33 and max(abs(a-b) for a,b in zip(yv,expected))<5.1e-10
traj=json.loads((R/'jobs/effort-gpt-6-luna-low-entropy-anomaly-20261011-191612/entropy-anomaly__6yBuZGL/agent/trajectory.json').read_text())
quote='The existing model gives a good calibration fit near friction 1.1, so the small-mass expansion is consistent with the records.'
assert quote in next(s['message'] for s in traj['steps'] if s['step_id']==12)
diag=json.loads((R/'results/trajectory-analysis-20261014/diagnostic-repairs.json').read_text())['cases'][0]
assert f"{100*max(diag['before'].values()):.2f}%"=='77.38%'
assert f"{100*max(diag['after'].values()):.4f}%"=='0.0443%'
wordcount=sum(len(re.findall(r'\b[\w’]+\b',n.split('Sources:')[0])) for n in notes[:12])
audit={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'final':str(final),'final_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),'slide_count':16,'titles':titles,'removed_original_slide':8,'new_apparatus_slide':3,'new_prompt_workflow_slide':4,'statistics_slides':[6,7,8],'detailed_trajectory_slide':9,'hint_setup_slide':10,'hint_results_slide':11,'hint_results_blank':True,'all_visible_fonts':dict(fontcounts),'original_workbooks_preserved_exactly':3,'new_analytic_charts':2,'analytic_profiles_verified_against_equations':True,'tables':checks,'recorded_agent_quote_verified':True,'diagnostic_values_verified':True,'main_seconds':600,'questions_seconds':180,'timing_seconds':timings,'main_note_word_count_including_timing_labels':wordcount,'visual_review':{'all_slides_inspected':True,'arrows_checked':'Left-to-right agent and action flow; bidirectional three-state contact graph','layout':'No unintended clipping or overlap in final full-size reimported previews','native_powerpoint_tested':False,'remaining_validator_warnings':'Conservative native table overflow estimates on backup slides 14 and 15. Final previews show table bottoms above their footnotes.'},'scientific_scope':'Recorded actions and source evidence only; no claim of access to private reasoning. Analyst repair does not count as a new model pass. Illustrative apparatus plots are equations at stated settings, not measured outcomes.'}
(B/'content-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({k:audit[k] for k in ['final_sha256','slide_count','main_seconds','questions_seconds','hint_results_blank','recorded_agent_quote_verified','analytic_profiles_verified_against_equations','original_workbooks_preserved_exactly','main_note_word_count_including_timing_labels']},indent=2))
