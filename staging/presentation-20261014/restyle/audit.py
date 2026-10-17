from pathlib import Path
import collections,hashlib,json,re,zipfile,xml.etree.ElementTree as E
R=Path('/Users/yp9846/Desktop/PhysLiteBench')
W=Path('/Users/yp9846/Documents/ChatGPT/phys-lite-bench')
B=W/'.presentation-restyle-20261014'
source=R/'output/presentation-20261014/PhysLiteBench-10-minute-draft.pptx'
final=W/'output/physlitebench-presentation/PhysLiteBench-10-minute-committee-style.pptx'
A='http://schemas.openxmlformats.org/drawingml/2006/main'
P='http://schemas.openxmlformats.org/presentationml/2006/main'
ns={'a':A,'p':P}
def content(z,p):return E.fromstring(z.read(p))
def texts(root):return [x.text or '' for x in root.findall('.//a:t',ns)]
def tablevalues(root):return [[texts(c) for c in row.findall('a:tc',ns)] for row in root.findall('.//a:tbl/a:tr',ns)]
checks=[]
with zipfile.ZipFile(source) as src,zipfile.ZipFile(final) as dst:
 for i in range(1,16):
  sp=f'ppt/slides/slide{i}.xml'; np=f'ppt/notesSlides/notesSlide{i}.xml'
  a,b=content(src,sp),content(dst,sp)
  assert tablevalues(a)==tablevalues(b),f'Table values changed slide{i}'
  assert texts(content(src,np))==texts(content(dst,np)),f'Notes changed slide{i}'
  def nums(root):
   return collections.Counter(re.findall(r'\d+(?:\.\d+)?', ' '.join(texts(root))))
  na,nb=nums(a),nums(b)
  if 2<=i<=9:
   na[str(i).zfill(2)]-=1;nb[str(i)]-=1
  assert +na==+nb,f'Visible numbers changed slide{i}: {na-nb}, {nb-na}'
  checks.append({'slide':i,'all_table_values_identical':True,'all_numeric_text_identical_except_page_zero_padding':True,'speaker_notes_identical':True})
 for p in src.namelist():
  if p.startswith('ppt/embeddings/'):
   assert src.read(p)==dst.read(p),f'Workbook changed {p}'
 hint=texts(content(dst,'ppt/slides/slide10.xml'))
 assert hint==['Hint comparison results','10','Pending completion'],hint
 notes=[texts(content(dst,f'ppt/notesSlides/notesSlide{i}.xml')) for i in range(1,16)]
 timings=[]
 for ts in notes:
  s=' '.join(ts);m=re.search(r'Target time: (\d+) seconds',s);timings.append(int(m.group(1)) if m else 0)
 assert sum(timings[:11])==600 and timings[11]==180,timings
 fonts=collections.Counter()
 for p in dst.namelist():
  if p.endswith('.xml') and p.startswith('ppt/'):
   root=E.fromstring(dst.read(p))
   for x in root.iter():
    if x.get('typeface') and not x.get('typeface').startswith('+'):fonts[x.get('typeface')]+=1
report={'source':str(source),'final':str(final),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'final_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),'slide_count':15,'font_counts':dict(fonts),'main_talk_seconds':600,'questions_seconds':180,'hint_results_blank':True,'exact_source_workbooks_preserved':3,'slide_checks':checks,'slide_title_edits':{'7':'A good fit can preserve the wrong physics','8':'Targeted corrections identify physical omissions','14':'Backup: collaborator results remain separate'},'scope':'Style-only edit; source findings, numeric content, table values, speaker notes, and source workbook bytes preserved.'}
(B/'content-preservation-audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['slide_checks','slide_title_edits']},indent=2))
style={'font':'Times New Roman','reference_deck':'/Users/yp9846/Desktop/thesis_committee_2026-10-05.pptx','reference_sha256':hashlib.sha256(Path('/Users/yp9846/Desktop/thesis_committee_2026-10-05.pptx').read_bytes()).hexdigest(),'guides':[{'url':'https://github.com/anthropics/skills/tree/main/skills/pptx','applied':'Native editable charts, readable type, contrast, spacing, render-and-review checks.'},{'url':'https://github.com/anthropics/skills/tree/main/skills/canvas-design','applied':'Visual hierarchy, restrained palette, whitespace, deliberate composition.'},{'url':'https://github.com/anthropics/skills/tree/main/skills/theme-factory','applied':'Consistent custom typography and color system based on the user-provided reference.'}],'design':{'slide_size_px':[1280,720],'title_font_px':48,'content_background':'#FFFFFF','text':'#111111','subtitle':'#444444','footer':'#777777','table_header':'#374151','table_band':'#F3F4F6','dark_cover':'#0B0B0F','accents':['#1D4E89','#0F766E','#B45309','#A50026','#5B4B7C'],'left_margin_px':48},'note':'The existing reference provides theme direction; no astronomical images are reused. Scientific values and existing source notes are preserved. The reserved hint-results slide remains blank.'}
(B/'style-provenance.json').write_text(json.dumps(style,indent=2))
