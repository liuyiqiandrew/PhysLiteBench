from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import time
import uuid

repo = Path('/Users/yp9846/Desktop/PhysLiteBench')
evidence = repo / 'results/reasoning-sweep-20261011'
private = Path('/private/tmp/physlite-prime-session')
sys.path.insert(0, str(repo / 'staging/reasoning-sweep-20261011/prime'))
from prime_catalog_probe import read_key
from openai import OpenAI, APIStatusError

hold_path = evidence / 'prime/provider-hold.json'
hold = json.loads(hold_path.read_text())
assert hold['state'] == 'provider_rate_limit_hold'
now = datetime.now(timezone.utc)
assert now >= datetime.fromisoformat(hold['probe_not_before_utc']), 'Cooldown has not elapsed'
ledger_path = private / 'ledger.json'
ledger = json.loads(ledger_path.read_text())
assert not any(row['status'] == 'pending' for row in ledger['requests'])
assert not any(json.loads(path.read_text())['status'] == 'running'
               and json.loads(path.read_text())['model'].startswith('deepseek/')
               for path in (evidence / 'runs').glob('*/run.json'))

payload = dict(model='deepseek/deepseek-v4.1-flash', input='Reply exactly OK.',
               reasoning={'effort': 'low'}, max_output_tokens=128000,
               extra_body={'usage': {'include': True}})
request_id = uuid.uuid4().hex
record = dict(id=request_id, purpose='Bounded availability probe after provider HTTP 429 cooldown; not a benchmark trial.',
              at=now.isoformat(), status='pending', reserved_usd=0.17,
              accounted_usd=0.17, transport_protocol='direct-diagnostic-probe',
              model=payload['model'], provider_reasoning_effort='low',
              max_output_tokens=128000,
              request_sha256=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
ledger['requests'].append(record)

def save_ledger():
    temporary = ledger_path.with_suffix('.probe.tmp')
    temporary.write_text(json.dumps(ledger, indent=2) + '\n')
    temporary.replace(ledger_path)

save_ledger()
key = read_key()
client = OpenAI(api_key=key, base_url='https://api.pinference.ai/api/v1', max_retries=0, timeout=45)
started = time.monotonic()
try:
    raw = client.responses.with_raw_response.create(**payload)
    response = raw.parse()
    usage = response.usage.model_dump() if response.usage else None
    record.update(http_status=raw.status_code, response_status=response.status,
                  output=response.output_text, rate_limit_headers={k: v for k, v in raw.headers.items()
                      if 'ratelimit' in k.lower() or k.lower() == 'retry-after'})
    if usage:
        input_tokens = usage.get('input_tokens', 0)
        output_tokens = usage.get('output_tokens', 0)
        cached = (usage.get('input_tokens_details') or {}).get('cached_tokens', 0) or 0
        cost = ((input_tokens - cached) * 0.30 + cached * 0.03 + output_tokens * 1.20) / 1e6
        reported = usage.get('cost') or 0
        record.update(status='completed', usage=usage, estimated_token_cost_usd=cost,
                      reported_cost_usd=reported, accounted_usd=max(cost, reported))
    else:
        record['status'] = 'missing_usage'
except APIStatusError as error:
    record.update(status='http_error', http_status=error.status_code,
                  error=error.response.text.replace(key, '[REDACTED]')[:1600],
                  rate_limit_headers={k: v for k, v in error.response.headers.items()
                      if 'ratelimit' in k.lower() or k.lower() == 'retry-after'})
except Exception as error:
    record.update(status='transport_error', error=str(error).replace(key, '[REDACTED]')[:800])
finally:
    client.close()
    record.update(finished_at=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic() - started)
    save_ledger()
    destination = evidence / 'prime' / ('provider-probe-' + now.strftime('%Y%m%d-%H%M%S') + '.json')
    destination.write_text(json.dumps(record, indent=2) + '\n')

success = record.get('response_status') == 'completed' and record.get('output', '').strip() == 'OK'
hold.setdefault('probes', []).append(str(destination.relative_to(evidence)))
if success:
    hold.update(state='released', released_utc=datetime.now(timezone.utc).isoformat(),
                successful_probe=str(destination.relative_to(evidence)))
else:
    from datetime import timedelta
    from email.utils import parsedate_to_datetime
    delay_seconds = 1200
    retry_after = record.get('rate_limit_headers', {}).get('retry-after')
    if retry_after:
        try:
            delay_seconds = max(delay_seconds, int(retry_after))
        except ValueError:
            try:
                delay_seconds = max(delay_seconds, (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError):
                pass
    hold['probe_not_before_utc'] = (datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)).isoformat()
hold_path.write_text(json.dumps(hold, indent=2) + '\n')
print(json.dumps({key: value for key, value in record.items() if key not in ['request_sha256', 'usage']}))
print(json.dumps(dict(benchmark_launches_released=success, next_probe_not_before=hold.get('probe_not_before_utc'))))
