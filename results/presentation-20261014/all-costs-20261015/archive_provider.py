from pathlib import Path
import json,hashlib
R=Path('/Users/yp9846/Desktop/PhysLiteBench');U=Path('/Users/yp9846/Documents/ChatGPT/phys-lite-bench/output/PhysLiteBench-all-cost-figures-20261015');S=U/'sources'
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
records=[];files={}
fields=['id','trial_tag','status','model','provider_reasoning_effort','response_id','usage','estimated_token_cost_usd','reported_cost_usd','accounted_usd','reserved_usd','delivered']
def archive(p,scope,model,effort,condition,trials):
 d=load(p);files[str(p.relative_to(R))]=sha(p)
 records.append({'source':str(p.relative_to(R)),'source_sha256':sha(p),'scope':scope,'model':model,'effort':effort,'condition':condition,'trials':trials,'requests':[{k:v for k,v in x.items() if k in fields} for x in d['requests']]})
d=load(S/'no-hint-results.json');ts=[t for t in d['trials'] if t['model'].startswith('deepseek')]
for rp in sorted({t['run_record'] for t in ts}):
 p=R/rp;r=load(p);files[rp]=sha(p);matching=[t['trial'] for t in ts if t['run_record']==rp]
 archive(R/r['prime_batch_ledger'],'no-hint-results.json',r['model'],r['reasoning_effort'],'baseline',matching)
d=load(S/'high-hint-results.json');ts=[t for t in d['trials'] if t['model'].startswith('deepseek')]
for t in ts:
 p=R/f"results/hint-comparison-20261014/blocks/{t['block']:03d}/{t['condition']}-prime-ledger.json"
 assert p.exists(),str(p)
 archive(p,'high-hint-results.json',t['model'],t['reasoning_effort'],t['condition'],[t['trial']])
assert len(records)==157
ids=[r['id'] for b in records for r in b['requests']];assert len(ids)==len(set(ids))
p=S/'provider-resource-records.json';p.write_text(json.dumps({'scope':'Allowlisted usage and cost metadata from saved per-run ledgers. No credentials or prompt bodies. This snapshot adds accounting evidence, not trials.','original_file_hashes':files,'batches':records},indent=2)+'\n')
m=load(U/'source-manifest.json');m.append({'file':'sources/provider-resource-records.json','source':'Saved benchmark Prime batch ledgers, paths and original SHA-256 hashes embedded','sha256':sha(p)});(U/'source-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps({'provider_batches':len(records),'unique_provider_requests':len(ids),'original_files_hashed':len(files)}))
