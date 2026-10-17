from pathlib import Path
import json,hashlib
R=Path('/Users/yp9846/Desktop/PhysLiteBench');S=R/'staging/hint-comparison-20261014';W=Path('/Users/yp9846/Documents/ChatGPT/phys-lite-bench/.parallel-hints-20261015')
for name in ['run_science_openai.py','run_science_prime.py']:
 s=(S/'tooling'/name).read_text()
 old='"-k", str(args.trials), "--job-name", name]'
 new='"-k", str(args.trials), "--job-name", name,\n                   "--extra-docker-compose", str(ROOT / "staging/hint-comparison-20261014/resources/eight-cpus.yaml")]'
 assert old in s
 (W/name).write_text(s.replace(old,new))
s=(S/'tooling/run_block.py').read_text()
old="""        for path in (RESULTS / 'blocks').glob('*/block.json'):
            assert json.loads(path.read_text())['status'] != 'running', 'Another block is active'
        folder.mkdir(parents=True)"""
new="""        active = [json.loads(path.read_text()) for path in (RESULTS / 'blocks').glob('*/block.json')
                  if json.loads(path.read_text())['status'] == 'running']
        assert len(active) < plan['max_concurrent_blocks'], 'Concurrent block limit reached'
        if block['model'].startswith('deepseek/'):
            assert not any(x['model'].startswith('deepseek/') for x in active), 'Another Prime block is active'
        assert not folder.exists(), 'Existing block must never be replaced'
        folder.mkdir(parents=True)"""
assert old in s;s=s.replace(old,new)
s=s.replace("'source_sha256': frozen['sha256'], 'plan_sha256': sha(RESULTS / 'plan.json')}","'source_sha256': frozen['sha256'], 'plan_sha256': sha(RESULTS / 'plan.json'),\n              'execution_cohort': 'parallel-eight-cpus-20261015',\n              'cpu_set': '0-7', 'max_concurrent_blocks': 2, 'max_concurrent_trials': 6}")
s=s.replace("summary = {**block, 'status': 'running', 'started_utc': now()}","summary = {**block, 'status': 'running', 'started_utc': now(),\n                   'execution_cohort': 'parallel-eight-cpus-20261015', 'cpu_set': '0-7',\n                   'max_concurrent_blocks': 2, 'max_concurrent_trials': 6}")
(W/'run_block.py').write_text(s)
(W/'eight-cpus.yaml').write_text('services:\n  main:\n    cpuset: "0-7"\n  harbor-docker-egress-control-sidecar:\n    cpuset: "0-7"\n')
# Keep task/prompt/grading and collector behavior unchanged. Provenance lives in each run record and the amendment.
