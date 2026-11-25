from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('run_slug')
args = parser.parse_args()
repo = Path('/Users/yp9846/Desktop/PhysLiteBench')
evidence = repo / 'results/reasoning-sweep-20261011'
run_path = evidence / 'runs' / args.run_slug / 'run.json'
run = json.loads(run_path.read_text())
ledger = json.loads(Path('/private/tmp/physlite-prime-session/ledger.json').read_text())
requests = ledger['requests'][run['prime_request_start_index']:run['prime_request_end_index']]
assert requests and not any(row['status'] == 'pending' for row in requests)
assert all(row['provider_reasoning_effort'] == run['reasoning_effort'] for row in requests)
snapshot = evidence / 'reviews' / (args.run_slug + '-unsettled-run.json')
batch_path = repo / run['prime_batch_ledger']
if not run['prime_requests_settled']:
    assert not snapshot.exists()
    shutil.copy2(run_path, snapshot)
    shutil.copy2(batch_path, evidence / 'reviews' / (args.run_slug + '-unsettled-ledger.json'))
batch = json.loads(batch_path.read_text())
batch['requests'] = requests
batch['reconciled_utc'] = datetime.now(timezone.utc).isoformat()
batch_path.write_text(json.dumps(batch, indent=2) + '\n')
run['prime_cost_range_usd'] = [sum(max(row.get('estimated_token_cost_usd') or 0, row.get('reported_cost_usd') or 0) for row in requests), sum(row['accounted_usd'] for row in requests)]
run['prime_requests_settled'] = True
run['prime_reconciled_utc'] = batch['reconciled_utc']
run_path.write_text(json.dumps(run, indent=2) + '\n')
print(json.dumps({'job': run['job'], 'requests': len(requests), 'cost_range_usd': run['prime_cost_range_usd'], 'settled': True}))
