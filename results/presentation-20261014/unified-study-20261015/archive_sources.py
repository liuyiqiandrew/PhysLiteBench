from pathlib import Path
import json,hashlib,shutil,urllib.request
W=Path('/Users/yp9846/Documents/ChatGPT/phys-lite-bench');R=Path('/Users/yp9846/Desktop/PhysLiteBench');O=W/'output/PhysLiteBench-study-figures-20261015';S=O/'sources';S.mkdir(parents=True,exist_ok=True)
manifest=[]
files={
 'no-hint-results.json':'results/reasoning-sweep-20261011/results.json',
 'high-hint-results.json':'results/hint-comparison-20261014/results.json',
 'trajectory-analysis.json':'results/trajectory-analysis-20261014/analysis.json',
 'no-hint-pricing.json':'results/reasoning-sweep-20261011/pricing.json',
 'high-hint-pricing.json':'results/hint-comparison-20261014/pricing.json',
 'no-hint-task-audit.json':'results/reasoning-sweep-20261011/task-source-audit.json',
 'high-final-audit.json':'results/hint-comparison-20261014/final-audit-20261015.json',
 'baseline-reuse.json':'results/hint-comparison-20261014/reused-baselines-20261015.json',
 'task-roster.json':'results/task-roster.json',
}
for label,batch in [('luna','luna-all-efforts-20261011-205240'),('sol-terra','sol-terra-low-medium-20261011-185155')]:
 for name in ['trials.csv','summary.csv','pricing.json','protocol.json','verification.json']:
  p=f'results/hint-comparison-20261014/imported-friend-results-20261015/{batch}/{name}'
  if (R/p).exists():files[f'baseline-{label}-{name}']=p
for name,rel in files.items():
 shutil.copy2(R/rel,S/name);manifest.append({'file':'sources/'+name,'source':str(R/rel),'sha256':hashlib.sha256((S/name).read_bytes()).hexdigest()})
commit='c925ddf30d7729a629125f942bab67d9edc53f1f'
for name in ['summary.csv','summary.md','pricing.json','verification.json']:
 url=f'https://raw.githubusercontent.com/liuyiqiandrew/PhysLiteBench/{commit}/results/hint-comparison-20261015-005800/{name}'
 raw=urllib.request.urlopen(url).read();p=S/('hint-matrix-'+name);p.write_bytes(raw)
 assert raw==(R/'results/presentation-20261014/collaborator-hints-20261015/sources'/name).read_bytes()
 manifest.append({'file':'sources/'+p.name,'source':url,'commit':commit,'sha256':hashlib.sha256(raw).hexdigest(),'verification_scope':'Published aggregate and verification metadata. Raw trajectories are not published.'})
# Preserve the original analytic chart coordinates used in the bead explanation.
import zipfile,xml.etree.ElementTree as E
z=zipfile.ZipFile(W/'.presentation-unified-20261015/source.pptx');ns={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
import posixpath
slide=E.fromstring(z.read('ppt/slides/slide16.xml')); relid=slide.find('.//c:chart',ns).get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
rels=E.fromstring(z.read('ppt/slides/_rels/slide16.xml.rels')); target=next(el.get('Target') for el in rels if el.get('Id')==relid)
root=E.fromstring(z.read(posixpath.normpath(posixpath.join('ppt/slides',target))))
series=[]
for se in root.findall('.//c:ser',ns):
 series.append({'x':[float(v.text) for v in se.findall('.//c:xVal//c:pt/c:v',ns)],'y':[float(v.text) for v in se.findall('.//c:yVal//c:pt/c:v',ns)]})
(S/'bead-analytic-series.json').write_text(json.dumps(series,indent=2)+'\n')
manifest.append({'file':'sources/bead-analytic-series.json','source':'Latest user-edited presentation, slide 16 analytic chart (not observed trajectories)','sha256':hashlib.sha256((S/'bead-analytic-series.json').read_bytes()).hexdigest()})
(O/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'bundle':str(O),'source_files':len(manifest),'bead_series':[len(s['x']) for s in series]}))
