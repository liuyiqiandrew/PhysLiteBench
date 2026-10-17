from pathlib import Path
import json, subprocess, sys
R = Path('/Users/yp9846/Desktop/PhysLiteBench')
E = R / 'results/reasoning-sweep-20261011'
s = json.loads(subprocess.check_output([sys.executable, str(R / 'staging/reasoning-sweep-20261011/tooling/status.py')]))
p = json.loads((E / 'prime/transport-v2-plan.json').read_text())
s['prime_next'] = [q for q in p['queue'] if not (E / 'runs' / f"prime-v2--deepseek-deepseek-v4.1-flash--{q['effort']}--{q['task']}" / 'run.json').exists()]
s['prime_remaining_batches'] = len(s['prime_next'])
s['prime_next'] = s['prime_next'][:1]
ledger=json.loads(Path('/private/tmp/physlite-prime-session/ledger.json').read_text())
s['prime_unsettled_requests']=sum(row['status']=='pending' for row in ledger['requests'])
selected=[json.loads(f.read_text()) for f in (E/'runs').glob('*/run.json')]
selected=[r for r in selected if not r['model'].startswith('deepseek/') or r.get('comparison_cohort')=='prime-transport-v2']
s['finished']=[{k:r[k] for k in ['task','model','reasoning_effort','passes','errors']} for r in selected if r['status'] in ['completed','completed_with_reviewed_errors']]
s['terminal_new_attempts']=sum(len(list((R/'jobs'/r['job']).glob('*/result.json'))) for r in selected)
hold=E/'prime/provider-hold.json'
s['prime_provider_hold']=json.loads(hold.read_text()) if hold.exists() and json.loads(hold.read_text()).get('state')=='provider_rate_limit_hold' else None
print(json.dumps(s))
