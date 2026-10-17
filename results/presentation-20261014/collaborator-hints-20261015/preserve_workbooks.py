"""Preserve the all four source chart workbooks despite slide and chart reordering."""
import copy,hashlib,json,posixpath,sys,zipfile,io,re
from pathlib import Path
import xml.etree.ElementTree as E
source,edited,output,receipt=map(Path,sys.argv[1:])
C='http://schemas.openxmlformats.org/drawingml/2006/chart';A='http://schemas.openxmlformats.org/drawingml/2006/main';R='http://schemas.openxmlformats.org/officeDocument/2006/relationships';CT='http://schemas.openxmlformats.org/package/2006/content-types'
for pre,url in [('c',C),('a',A),('r',R)]:E.register_namespace(pre,url)
ns={'c':C}
with zipfile.ZipFile(source) as z:src={i.filename:z.read(i) for i in z.infolist()}
with zipfile.ZipFile(edited) as z:dst={i.filename:z.read(i) for i in z.infolist()}
def charts(d):return {p:E.fromstring(b) for p,b in d.items() if '/charts/' in p and p.endswith('.xml') and '/_rels/' not in p}
def fingerprint(root):
 def value(tag, text):
  return float(text) if tag in ['val','xVal','yVal','bubbleSize'] else text
 return [[(tag,[value(tag,v.text) for v in (ser.find('c:'+tag,ns).findall('.//c:v',ns) if ser.find('c:'+tag,ns) is not None else [])]) for tag in ['tx','cat','val','xVal','yVal','bubbleSize']] for ser in root.findall('.//c:ser',ns)]
newcharts=charts(dst);records=[]
for cp,old in charts(src).items():
 matches=[(p,v) for p,v in newcharts.items() if fingerprint(v)==fingerprint(old)]
 if len(matches)!=1:raise ValueError(f'Unchanged chart not uniquely identified: {cp}, {len(matches)}')
 np,new=matches[0]
 for a,b in zip(old.findall('.//c:ser',ns),new.findall('.//c:ser',ns)):
  for tag in ['tx','cat','val','xVal','yVal','bubbleSize']:
   x=a.find('c:'+tag,ns);y=b.find('c:'+tag,ns)
   if x is not None:
    i=list(b).index(y);b.remove(y);b.insert(i,copy.deepcopy(x))
 ext=old.find('c:externalData',ns)
 if ext is None:raise ValueError('Original chart workbook missing')
 oe=new.find('c:externalData',ns)
 if oe is not None:new.remove(oe)
 new.append(copy.deepcopy(ext))
 sr=posixpath.join(posixpath.dirname(cp),'_rels',posixpath.basename(cp)+'.rels')
 dr=posixpath.join(posixpath.dirname(np),'_rels',posixpath.basename(np)+'.rels')
 if dr in dst:raise ValueError('Unexpected existing chart relationship')
 dst[dr]=src[sr]
 for rel in E.fromstring(src[sr]):
  if rel.get('Type')!=R+'/package':raise ValueError('Unexpected source relationship')
  target=posixpath.normpath(posixpath.join(posixpath.dirname(cp),rel.get('Target')))
  desttarget=posixpath.normpath(posixpath.join(posixpath.dirname(np),rel.get('Target')))
  dst[desttarget]=src[target]
  records.append({'source_chart':cp,'edited_chart':np,'workbook':desttarget,'sha256':hashlib.sha256(src[target]).hexdigest(),'exact_bytes_preserved':True})
 # Refresh only numeric cache text from the exact preserved workbook cells.
 # PowerPoint may rewrite cache decimal spellings without changing worksheet values.
 with zipfile.ZipFile(io.BytesIO(src[target])) as wb:
  sn={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
  ws=E.fromstring(wb.read('xl/worksheets/sheet1.xml'))
  cells={cell.get('r'):cell.find('s:v',sn).text for cell in ws.findall('.//s:c',sn) if cell.find('s:v',sn) is not None}
  refresh=0
  for ref in new.findall('.//c:numRef',ns):
   formula=ref.find('c:f',ns).text
   m=re.fullmatch(r"'Chart Data'!\$([A-Z]+)\$(\d+):\$([A-Z]+)\$(\d+)",formula)
   if not m or m[1]!=m[3]:raise ValueError('Unsupported original range: '+formula)
   points=ref.findall('c:numCache/c:pt',ns)
   for pt in points:
    address=m[1]+str(int(m[2])+int(pt.get('idx')))
    value=pt.find('c:v',ns);exact=cells[address]
    if float(value.text)!=float(exact):raise ValueError('Source cache and workbook values differ numerically')
    if value.text!=exact:value.text=exact;refresh+=1
  records[-1]['numeric_cache_spellings_refreshed_from_original_workbook']=refresh
 dst[np]=E.tostring(new,encoding='utf-8',xml_declaration=True)
# Preserve the requested font in all inherited master font defaults.
for name,data in list(dst.items()):
 if '/theme/' in name and name.endswith('.xml'):
  root=E.fromstring(data);changed=False
  for font in root.findall(f'.//{{{A}}}fontScheme/{{{A}}}majorFont/{{{A}}}latin')+root.findall(f'.//{{{A}}}fontScheme/{{{A}}}minorFont/{{{A}}}latin'):
   if font.get('typeface')!='Times New Roman':font.set('typeface','Times New Roman');changed=True
  if changed:dst[name]=E.tostring(root,encoding='utf-8',xml_declaration=True)
oldct=E.fromstring(src['[Content_Types].xml']);newct=E.fromstring(dst['[Content_Types].xml'])
for el in oldct:
 if el.tag==f'{{{CT}}}Default' and el.get('Extension')=='xlsx' and not any(x.get('Extension')=='xlsx' for x in newct):newct.append(copy.deepcopy(el))
 if el.tag==f'{{{CT}}}Override' and '/embeddings/' in el.get('PartName','') and not any(x.get('PartName')==el.get('PartName') for x in newct):newct.append(copy.deepcopy(el))
E.register_namespace('',CT);dst['[Content_Types].xml']=E.tostring(newct,encoding='utf-8',xml_declaration=True)
with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
 for p,data in dst.items():z.writestr(p,data)
receipt.write_text(json.dumps({'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'records':records,'new_charts':'Two new collaborator charts use complete literal source data. Four original chart workbooks are preserved exactly; numeric XML formatting differences are normalized only for matching.'},indent=2))
print(f'Preserved {len(records)} original workbooks and their chart references')
