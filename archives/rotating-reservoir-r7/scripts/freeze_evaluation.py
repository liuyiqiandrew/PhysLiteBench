"""Record a source-frozen three-trial evaluation before launch."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]

def hashes(directory):
    return {str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob('*')) if p.is_file()
            and not any(s in ('__pycache__','.pytest_cache','.git') for s in p.parts)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task')
    parser.add_argument('--revision',type=int,required=True)
    args=parser.parse_args()
    task=ROOT/'tasks'/args.task
    destination=ROOT/'results'/f'{args.task}-r{args.revision}-evaluation-plan.json'
    if destination.exists():
        old=json.loads(destination.read_text())
        assert old['source_sha256']==hashes(task), 'Frozen revision changed after its evaluation plan'
        print(destination)
        return
    plan=dict(task=args.task,revision=args.revision,created_at=datetime.now(timezone.utc).isoformat(),
              base_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              model='gpt-5.6-luna',reasoning_effort='high',trials=3,concurrency=3,physics_hint=False,
              agent_seconds=600,verifier_seconds=60,setup_timeout_multiplier=3,
              target_passes=0,minimum_reviewed_physical_failures=2,
              unchanged_task_resampling=False,source_sha256=hashes(task),
              qualification='All three trials retained and reviewed; controls alone do not establish difficulty.')
    destination.write_text(json.dumps(plan,indent=2)+'\n')
    print(destination)

if __name__=='__main__':main()
