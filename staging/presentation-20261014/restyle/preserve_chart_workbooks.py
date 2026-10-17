"""Restore unchanged source chart data links and workbook bytes after first-party style editing."""
import copy, hashlib, json, posixpath, sys, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
source, edited, output, receipt = map(Path, sys.argv[1:])
C='http://schemas.openxmlformats.org/drawingml/2006/chart'
A='http://schemas.openxmlformats.org/drawingml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PKG='http://schemas.openxmlformats.org/package/2006/relationships'
CT='http://schemas.openxmlformats.org/package/2006/content-types'
ET.register_namespace('c',C); ET.register_namespace('a',A); ET.register_namespace('r',R)
ns={'c':C}
with zipfile.ZipFile(source) as z: src={i.filename:z.read(i) for i in z.infolist()}
with zipfile.ZipFile(edited) as z: dst={i.filename:z.read(i) for i in z.infolist()}
contracts=[]
for cp in sorted(k for k in src if '/charts/' in k and k.endswith('.xml') and '/_rels/' not in k):
 if cp not in dst: raise ValueError('Missing edited chart '+cp)
 old=ET.fromstring(src[cp]); new=ET.fromstring(dst[cp])
 oldser=old.findall('.//c:ser',ns); newser=new.findall('.//c:ser',ns)
 if len(oldser)!=len(newser):raise ValueError('Series count changed')
 for a,b in zip(oldser,newser):
  for tag in ['tx','cat','val','xVal','yVal','bubbleSize']:
   x=a.find(f'c:{tag}',ns); y=b.find(f'c:{tag}',ns)
   if x is None and y is None:continue
   if x is None or y is None:raise ValueError(f'Missing data element {cp}/{tag}')
   xv=[q.text for q in x.findall('.//c:v',ns)]; yv=[q.text for q in y.findall('.//c:v',ns)]
   if xv!=yv:raise ValueError(f'Chart data changed {cp}/{tag}: {xv} != {yv}')
   index=list(b).index(y);b.remove(y);b.insert(index,copy.deepcopy(x))
 external=old.find('c:externalData',ns)
 if external is None:raise ValueError('Source has no workbook')
 existing=new.find('c:externalData',ns)
 if existing is not None:new.remove(existing)
 new.append(copy.deepcopy(external))
 relpath=posixpath.join(posixpath.dirname(cp),'_rels',posixpath.basename(cp)+'.rels')
 relroot=ET.fromstring(src[relpath])
 if relpath in dst and dst[relpath]!=src[relpath]:raise ValueError('Unexpected relationships to reconcile')
 dst[relpath]=src[relpath]
 for rel in relroot:
  if rel.get('Type')!=R+'/package':raise ValueError('Unexpected relationship')
  target=posixpath.normpath(posixpath.join(posixpath.dirname(cp),rel.get('Target')))
  dst[target]=src[target]
  contracts.append({'chart':cp,'workbook':target,'workbook_sha256':hashlib.sha256(src[target]).hexdigest(),'preserved_exact_bytes':True,'series_count':len(oldser)})
 # The first-party export retained the imported chart-space background.
 for color in new.findall(f'./c:spPr/{{{A}}}solidFill/{{{A}}}srgbClr',ns):
  color.set('val','FFFFFF')
 dst[cp]=ET.tostring(new,encoding='utf-8',xml_declaration=True)
# Apply the requested font to inherited master font defaults as well.
for name,data in list(dst.items()):
 if '/theme/' in name and name.endswith('.xml'):
  root=ET.fromstring(data);changed=False
  for font in root.findall(f'.//{{{A}}}fontScheme/{{{A}}}majorFont/{{{A}}}latin')+root.findall(f'.//{{{A}}}fontScheme/{{{A}}}minorFont/{{{A}}}latin'):
   if font.get('typeface')!='Times New Roman':font.set('typeface','Times New Roman');changed=True
  if changed:dst[name]=ET.tostring(root,encoding='utf-8',xml_declaration=True)
oldct=ET.fromstring(src['[Content_Types].xml']); newct=ET.fromstring(dst['[Content_Types].xml'])
for el in oldct:
 if el.tag==f'{{{CT}}}Default' and el.get('Extension')=='xlsx':
  if not any(x.get('Extension')=='xlsx' for x in newct):newct.append(copy.deepcopy(el))
 if el.tag==f'{{{CT}}}Override' and '/embeddings/' in el.get('PartName',''):
  if not any(x.get('PartName')==el.get('PartName') for x in newct):newct.append(copy.deepcopy(el))
ET.register_namespace('',CT)
dst['[Content_Types].xml']=ET.tostring(newct,encoding='utf-8',xml_declaration=True)
with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
 for name,data in dst.items():z.writestr(name,data)
receipt.write_text(json.dumps({'reason':'Preserve unchanged original workbook snapshots and source chart references after style-only import/export. No workbook regeneration or data editing.','source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'contracts':contracts},indent=2))
print(f'Preserved {len(contracts)} exact source workbooks and all data references')
