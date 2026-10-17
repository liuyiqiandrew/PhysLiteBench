from pathlib import Path
import json,re,urllib.request,urllib.error,zipfile,xml.etree.ElementTree as ET

DOC=Path('/Users/yp9846/Desktop/api.docx')
def read_key():
    with zipfile.ZipFile(DOC) as archive:
        root=ET.fromstring(archive.read('word/document.xml'))
    text=''.join(root.itertext())
    keys=set(re.findall(r'pit_[A-Za-z0-9_-]{20,}',text))
    if len(keys)!=1:
        raise ValueError(f'Expected one Prime key; found {len(keys)}. No document content displayed.')
    return keys.pop()

if __name__=='__main__':
    key=read_key()
    req=urllib.request.Request('https://api.pinference.ai/api/v1/models',headers={'Authorization':'Bearer '+key})
    try:
        with urllib.request.urlopen(req,timeout=40) as response:
            data=json.load(response)
    except urllib.error.HTTPError as e:
        detail=e.read().decode(errors='replace').replace(key,'[REDACTED]')
        print(json.dumps({'authenticated':False,'http_status':e.code,'detail':detail[:1500],'server':e.headers.get('Server')}));raise SystemExit(1)
    safe=json.dumps(data).replace(key,'[REDACTED]')
    Path('/private/tmp/prime-model-catalog.json').write_text(safe+'\n')
    rows=data.get('data',data.get('models',[])) if isinstance(data,dict) else data
    selected=[row for row in rows if 'deepseek' in json.dumps(row).lower()]
    print(json.dumps({'authenticated':True,'total_models':len(rows),'deepseek_models':selected},indent=2))
