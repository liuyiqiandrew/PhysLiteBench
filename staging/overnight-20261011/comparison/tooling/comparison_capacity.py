from pathlib import Path
import json
root=Path('/Users/yp9846/Desktop/PhysLiteBench')
c=root/'staging/overnight-20261011/comparison'
remaining=0
for p in (c/'runs').glob('*/run.json'):
    r=json.loads(p.read_text())
    if r['status']=='running':
        terminal=len(list((root/'jobs'/r['job']).glob('*/result.json')))
        remaining+=max(0,r['trials_requested']-terminal)
print(remaining)
