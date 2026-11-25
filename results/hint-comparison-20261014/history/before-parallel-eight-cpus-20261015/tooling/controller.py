"""Advance the fixed experiment until completion or an error requires review."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import subprocess
import sys

ROOT = Path('/Users/yp9846/Desktop/PhysLiteBench')
NAME = 'hint-comparison-20261014'
STAGING = ROOT / 'staging' / NAME
RESULTS = ROOT / 'results' / NAME


def main():
    plan = json.loads((RESULTS / 'plan.json').read_text())
    env = os.environ.copy()
    env.update(UV_CACHE_DIR='/private/tmp/physlitebench-inspection-uv', UV_PYTHON_INSTALL_DIR=str(ROOT / '.cache/python'))
    for block in plan['blocks']:
        path = RESULTS / 'blocks' / f"{block['index']:03d}" / 'block.json'
        if path.exists():
            state = json.loads(path.read_text())['status']
            if state == 'completed':
                continue
            print(json.dumps({'stopped': 'Existing block requires review', 'index': block['index'], 'state': state}), flush=True)
            return
        print(json.dumps({'launch_block': block['index'], 'model': block['model'], 'task': block['task'], 'repeat': block['repeat']}), flush=True)
        result = subprocess.run([sys.executable, str(STAGING / 'tooling/run_block.py'), str(block['index'])], cwd=ROOT, env=env)
        subprocess.run([sys.executable, str(STAGING / 'tooling/collect.py')], cwd=ROOT, env=env, check=True)
        command = [str(ROOT / '.cache/bin/uv'), 'run', '--offline', '--python', '3.13', '--with', 'matplotlib==3.10.7', '--with', 'numpy==2.5.4', 'python', str(STAGING / 'tooling/plot.py')]
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        if result.returncode or not path.exists() or json.loads(path.read_text())['status'] != 'completed':
            print(json.dumps({'stopped': 'Review required; original outcomes preserved', 'index': block['index']}), flush=True)
            return
    print(json.dumps({'completed': True, 'attempts': plan['expected_attempts'], 'at': datetime.now(timezone.utc).isoformat()}), flush=True)


if __name__ == '__main__':
    main()
