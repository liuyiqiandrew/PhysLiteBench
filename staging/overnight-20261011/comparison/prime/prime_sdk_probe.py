from pathlib import Path
from prime_catalog_probe import read_key
from openai import OpenAI,APIStatusError
import json
key=read_key()
client=OpenAI(api_key=key,base_url='https://api.pinference.ai/api/v1',max_retries=0,timeout=40)
try:
    data=client.models.list().model_dump()
except APIStatusError as e:
    print(json.dumps({'status':e.status_code,'detail':str(e).replace(key,'[REDACTED]')[:1000]}));raise SystemExit(1)
Path('/private/tmp/prime-model-catalog.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({'authenticated':True,'deepseek_models':[x for x in data['data'] if 'deepseek' in json.dumps(x).lower()]},indent=2))
