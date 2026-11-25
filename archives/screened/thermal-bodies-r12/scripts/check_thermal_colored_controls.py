"""Run the unchanged public/private harness for both thermal r12 controls."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'tasks/thermal-bodies'
report={}
for label,source in [('oracle',TASK/'solution/model.py'),('shortcut',ROOT/'scripts/thermal_colored_baseline.py')]:
    with tempfile.TemporaryDirectory(prefix='thermal-r12-'+label+'-') as tmp:
        root=Path(tmp);shutil.copytree(TASK/'environment',root/'app');shutil.copytree(TASK/'tests',root/'tests')
        shutil.copy2(source,root/'app/model.py')
        env=dict(os.environ,PYTHONPATH=str(root/'app'),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        start=time.time()
        result=subprocess.run([sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',str(root/'app/test_public.py'),str(root/'tests/test_hidden.py')],cwd=root,env=env,capture_output=True,text=True,timeout=60)
        report[label]=dict(returncode=result.returncode,seconds=time.time()-start,stdout=result.stdout,stderr=result.stderr)
assert report['oracle']['returncode']==0 and report['shortcut']['returncode']==1
assert '3 failed, 4 passed' in report['shortcut']['stdout']
(ROOT/'results/thermal-r12-local-controls.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
