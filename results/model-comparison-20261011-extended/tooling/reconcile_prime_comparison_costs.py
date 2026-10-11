"""Include provider requests that outlive an agent timeout in panel costs."""
from pathlib import Path
from datetime import datetime
import json,statistics
R=Path('/Users/yp9846/Desktop/PhysLiteBench');C=R/'staging/overnight-20261011/comparison';model='deepseek/deepseek-v4.1-flash'
data=json.loads((C/'results.json').read_text());ledger=json.loads(Path('/private/tmp/physlite-prime-session/ledger.json').read_text())
runs=[json.loads(p.read_text()) for p in (C/'runs').glob('*/run.json')];runs=[r for r in runs if r['model']==model]
if not runs:raise SystemExit('No Prime benchmark runs yet')
start=min(datetime.fromisoformat(r['started_at']) for r in runs)
requests=[r for r in ledger['requests'] if datetime.fromisoformat(r['at'])>=start]
rows=[r for r in data['trials'] if r['model']==model];events={e['response_id']:e for r in rows for e in r['usage_events']}
known=[r for r in requests if r.get('usage')];unknown=[r for r in requests if not r.get('usage')]
response_ids={r.get('response_id') for r in known};assert set(events)<=response_ids, 'Native token events missing from provider ledger'
for r in known:
 if r.get('response_id') in events:
  assert abs(events[r['response_id']]['estimated_standard_api_usd']-r['estimated_token_cost_usd'])<1e-10
lower=sum(r['estimated_token_cost_usd'] for r in known);upper=lower+sum(r['accounted_usd'] for r in unknown)
summary=next(x for x in data['summary'] if x['model']==model);n=summary['attempts']
summary['native_logged_total_standard_api_usd']=summary['total_standard_api_usd'];summary['total_standard_api_usd']=upper;summary['mean_standard_api_usd']=upper/n
summary['prime_cost_range_usd']=[lower,upper];summary['mean_cost_range_usd']=[lower/n,upper/n];summary['prime_unknown_usage_requests']=len(unknown)
summary['cost_basis']='Prime catalog token-cost estimate including completed provider requests after agent timeout; conservative reserved maximum substituted for any request lacking usage.'
summary['prime_extra_requests_without_native_usage']=[dict(id=r['id'],response_id=r.get('response_id'),status=r['status'],delivered=r.get('delivered'),estimated_usd=r.get('estimated_token_cost_usd'),reserved_usd=r.get('reserved_usd')) for r in requests if r.get('response_id') not in events]
data['prime_cost_audit']=dict(benchmark_cutoff=start.isoformat(),basis='All requests after first Prime benchmark batch started; no compatibility calls or other Prime workloads were launched after that cutoff.',requests=len(requests),completed_usage_requests=len(known),unknown_usage_requests=len(unknown),native_response_ids_matched=len(events),observed_cost_usd=lower,conservative_cost_upper_usd=upper,compatibility_cost_excluded=True)
latencies=[(datetime.fromisoformat(r['finished_at'])-datetime.fromisoformat(r['at'])).total_seconds() for r in requests if r.get('finished_at')]
if latencies:
 data['prime_cost_audit']['provider_request_duration_seconds']=dict(median=statistics.median(latencies),maximum=max(latencies),above_600_seconds=sum(x>600 for x in latencies))
data['prime_cost_audit']['responses_after_client_disconnect']=sum(r.get('delivered') is False for r in requests)
(C/'results.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data['prime_cost_audit']))
